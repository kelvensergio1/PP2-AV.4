import sqlite3

DB_NAME = "interacoes.db"

def criar_tabela():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interacoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT,
            nome_arquivo TEXT,
            predicao TEXT,
            probabilidade REAL
        )
    """)
    conn.commit()
    conn.close()

def registrar_interacao(data_hora, nome_arquivo, predicao, prob):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO interacoes (data_hora, nome_arquivo, predicao, probabilidade)
        VALUES (?, ?, ?, ?)
    """, (data_hora, nome_arquivo, predicao, prob))
    conn.commit()
    conn.close()
