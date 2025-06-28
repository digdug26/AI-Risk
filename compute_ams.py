from dotenv import load_dotenv
load_dotenv(".env.local")

import os
import logging
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import create_engine, text


EMPLOYMENT_CSV = os.path.join('data', 'oes_employment_2024.csv')
FIRM_CSV = os.path.join('data', 'qcew_firm_counts.csv')
OUTPUT_CSV = os.path.join('reports', 'ams_latest.csv')

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def load_base_tables():
    emp = pd.read_csv(EMPLOYMENT_CSV, dtype={'soc': str})
    if 'employment_base' in emp.columns:
        emp = emp.rename(columns={'employment_base': 'employment'})
    if 'employment' not in emp.columns:
        raise ValueError('employment column missing in oes_employment_2024.csv')

    firm = pd.read_csv(FIRM_CSV, dtype={'soc': str})
    if 'firm_base' in firm.columns:
        firm = firm.rename(columns={'firm_base': 'firms'})
    if 'firms' not in firm.columns:
        raise ValueError('firms column missing in qcew_firm_counts.csv')
    return emp[['soc', 'employment']], firm[['soc', 'firms']]


def fetch_soc_events(conn, start_date):
    query = text(
        """
        SELECT se.soc, e.company_id, se.weighted_headcount, se.p_ai_causal
        FROM soc_events se
        JOIN events e ON e.event_id = se.event_id
        WHERE e.pub_date >= :start
        """
    )
    return pd.read_sql(query, conn, params={'start': start_date})


def compute_ams(df, emp, firm):
    df['headcount_adj'] = df['weighted_headcount'] * df['p_ai_causal']
    df['mention_flag'] = df['p_ai_causal'] >= 0.7

    by_comp = df.groupby(['soc', 'company_id']).agg(
        headcount_adj=('headcount_adj', 'sum'),
        mention_cnt=('mention_flag', 'sum'),
    ).reset_index()

    by_soc = by_comp.groupby('soc').agg(
        H_s=('headcount_adj', 'sum'),
        C_s=('mention_cnt', lambda x: (x > 0).sum()),
    ).reset_index()

    res = by_soc.merge(emp, on='soc', how='left').merge(firm, on='soc', how='left')

    missing_emp = res['employment'].isna()
    for soc in res.loc[missing_emp, 'soc']:
        logging.warning('Missing employment_base for SOC %s; skipping.', soc)
    res = res[~missing_emp]

    missing_firm = res['firms'].isna()
    for soc in res.loc[missing_firm, 'soc']:
        logging.warning('Missing firm_base for SOC %s; skipping.', soc)
    res = res[~missing_firm]

    res['r1'] = (res['H_s'] / res['employment']).clip(upper=0.10)
    res['r2'] = res['C_s'] / res['firms']

    min_r1 = res['r1'].min()
    max_r1 = res['r1'].max()
    min_r2 = res['r2'].min()
    max_r2 = res['r2'].max()

    res['z1'] = (res['r1'] - min_r1) / (max_r1 - min_r1) if max_r1 > min_r1 else 0
    res['z2'] = (res['r2'] - min_r2) / (max_r2 - min_r2) if max_r2 > min_r2 else 0

    res['ams_raw'] = 0.5 * res['z1'] + 0.5 * res['z2']
    res['ams'] = res['ams_raw'].pow(0.5)

    res['min_r1'] = min_r1
    res['max_r1'] = max_r1
    res['min_r2'] = min_r2
    res['max_r2'] = max_r2

    return res


def persist_results(conn, df, run_date):
    insert_sql = text(
        """
        INSERT INTO ai_adoption_scores (
            soc, run_date, r1, r2, z1, z2, ams_raw, ams,
            min_r1, max_r1, min_r2, max_r2
        ) VALUES (
            :soc, :run_date, :r1, :r2, :z1, :z2, :ams_raw, :ams,
            :min_r1, :max_r1, :min_r2, :max_r2
        ) ON CONFLICT (soc) DO UPDATE SET
            run_date = EXCLUDED.run_date,
            r1 = EXCLUDED.r1,
            r2 = EXCLUDED.r2,
            z1 = EXCLUDED.z1,
            z2 = EXCLUDED.z2,
            ams_raw = EXCLUDED.ams_raw,
            ams = EXCLUDED.ams,
            min_r1 = EXCLUDED.min_r1,
            max_r1 = EXCLUDED.max_r1,
            min_r2 = EXCLUDED.min_r2,
            max_r2 = EXCLUDED.max_r2
        """
    )
    records = df.to_dict(orient='records')
    for rec in records:
        rec['run_date'] = run_date
    with conn.begin():
        conn.execute(insert_sql, records)


def log_movers(df_new, df_prev):
    if df_prev is None or df_prev.empty:
        logging.info('No prior run found; skipping movers log.')
        return
    merged = df_new[['soc', 'ams']].merge(df_prev, on='soc', how='left', suffixes=('_new', '_old'))
    merged['delta'] = merged['ams_new'] - merged['ams_old']
    movers = merged.sort_values('delta', ascending=False).head(10)
    logging.info('Top 10 \u2191-AMS movers vs prior run:')
    for _, row in movers.iterrows():
        logging.info('%s %+0.4f', row['soc'], row['delta'])


def main():
    pg_uri = os.environ.get('PG_URI')
    if not pg_uri:
        logging.error('PG_URI env var required')
        return 1
    engine = create_engine(pg_uri)
    run_date = date.today()
    start_date = run_date - timedelta(days=365)

    emp, firm = load_base_tables()

    with engine.connect() as conn:
        events = fetch_soc_events(conn, start_date)
        prev = pd.read_sql('SELECT soc, ams FROM ai_adoption_scores', conn)
        result = compute_ams(events, emp, firm)
        log_movers(result, prev)
        persist_results(conn, result, run_date)
        result.to_csv(OUTPUT_CSV, index=False)
        logging.info('Wrote %d SOC rows to %s', len(result), OUTPUT_CSV)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
