try:
    import psycopg2
except Exception:  # pragma: no cover - allow tests without psycopg2
    psycopg2 = None

def get_company_id(conn, name):
    """Return company_id for given name, inserting if needed."""
    cur = conn.cursor()
    cur.execute("SELECT company_id FROM companies WHERE name=%s", (name,))
    row = cur.fetchone()
    if row:
        cur.close()
        return row[0]
    cur.execute("INSERT INTO companies (name) VALUES (%s) RETURNING company_id", (name,))
    cid = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return cid
