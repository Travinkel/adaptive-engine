# adaptive_engine/ncde/state_manager.py

import psycopg2
import psycopg2.pool

class StateManager:
    """
    Manages persistence and retrieval of cognitive state vectors.
    """
    def __init__(self, db_config):
        self.pool = psycopg2.pool.SimpleConnectionPool(1, 10, **db_config)

    def get_state(self, learner_id):
        """
        Retrieves the latest cognitive state for a given learner.
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT state_vector FROM cognitive_states WHERE learner_id = %s ORDER BY timestamp DESC LIMIT 1", (learner_id,))
                result = cur.fetchone()
                return result[0] if result else None
        finally:
            self.pool.putconn(conn)

    def save_state(self, learner_id, state_vector):
        """
        Saves the cognitive state for a given learner.
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO cognitive_states (learner_id, state_vector) VALUES (%s, %s)", (learner_id, state_vector))
            conn.commit()
        finally:
            self.pool.putconn(conn)
