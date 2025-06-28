from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(BASE_DIR, 'compute_ams.py')

default_args = {
    'depends_on_past': False,
}

dag = DAG(
    dag_id='compute_ai_adoption_scores',
    default_args=default_args,
    schedule_interval='0 8 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

BashOperator(
    task_id='run_compute_ams',
    bash_command=f'python {SCRIPT}',
    dag=dag,
)
