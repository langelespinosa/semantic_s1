import math
import os
import warnings

import faiss
import mysql.connector
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from mysql.connector import Error
from sentence_transformers import SentenceTransformer
import uvicorn

# Cargar las variables de entorno para Render
MASTER_KEY = os.getenv("MASTER_KEY")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "db4free.net"),
    "user": os.getenv("DB_USER", "practice"),
    "password": os.getenv("DB_PASSWORD", "BAg0T?@EspLj"),
    "database": os.getenv("DB_DATABASE", "email_practice"),
    "port": int(os.getenv("DB_PORT", 3306))
}

warnings.filterwarnings("ignore", category=FutureWarning)
app = FastAPI(title="API Búsqueda Usuarios & Aliases", version="1.4.0")

def obtener_usuarios_desde_mysql():
    connection, cursor = None, None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT u.*, t.id AS transport_id, t.domain, t.transport
        FROM users u
        JOIN transports t ON u.dominio = t.id
        WHERE u.deleted_at IS NULL AND t.deleted_at IS NULL;
        """
        cursor.execute(query)
        return cursor.fetchall()
    except Error as e:
        print(f"❌ Error al conectar con MySQL (users): {e}")
        return []
    finally:
        if cursor: cursor.close()
        if connection and connection.is_connected():
            connection.close()

def obtener_aliases_desde_mysql():
    connection, cursor = None, None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT * FROM aliases
        WHERE deleted_at IS NULL;
        """
        cursor.execute(query)
        return cursor.fetchall()
    except Error as e:
        print(f"❌ Error al conectar con MySQL (aliases): {e}")
        return []
    finally:
        if cursor: cursor.close()
        if connection and connection.is_connected():
            connection.close()

print("🔄 Cargando usuarios desde MySQL...")
usuarios = obtener_usuarios_desde_mysql()
if not usuarios:
    raise RuntimeError("No se encontraron usuarios en la base de datos")

corpus_users, id_map_users = [], {}
for i, usuario in enumerate(usuarios):
    text = (
        f"ID: {usuario['id']} Login: {usuario['login']} Email: {usuario['email']} "
        f"Maildir: {usuario['maildir']} Identificacion: {usuario['identificacion']} "
        f"Grupo: {usuario['grupo']} Dominio: {usuario['domain']} Transport: {usuario['transport']}"
    )
    corpus_users.append(text)
    id_map_users[i] = usuario["id"]

print("🔄 Cargando aliases desde MySQL...")
aliases = obtener_aliases_desde_mysql()
if not aliases:
    raise RuntimeError("No se encontraron aliases en la base de datos")

corpus_aliases, id_map_aliases = [], {}
for i, alias in enumerate(aliases):
    text = f"ID: {alias['id']} Local: {alias['local']} Remoto: {alias['remoto']}"
    corpus_aliases.append(text)
    id_map_aliases[i] = alias["id"]

print("🔄 Generando embeddings...")
model = SentenceTransformer('multi-qa-distilbert-cos-v1')

embeddings_users = model.encode(corpus_users, normalize_embeddings=True)
index_users = faiss.IndexFlatIP(embeddings_users.shape[1])
index_users.add(np.array(embeddings_users, dtype=np.float32))

embeddings_aliases = model.encode(corpus_aliases, normalize_embeddings=True)
index_aliases = faiss.IndexFlatIP(embeddings_aliases.shape[1])
index_aliases.add(np.array(embeddings_aliases, dtype=np.float32))

print("✅ Índices FAISS creados exitosamente")

def buscar(index, id_map, docs, query, threshold=0.3):
    query_vec = model.encode([query], normalize_embeddings=True)
    total = len(docs)
    D, I = index.search(np.array(query_vec, dtype=np.float32), total)
    resultados = []
    for score, idx in zip(D[0], I[0]):
        if score >= threshold:
            resultados.append((id_map[idx], float(score)))
    resultados.sort(key=lambda x: x[1], reverse=True)
    return resultados

def buscar_en_users_sql(palabra_clave):
    termino = f"%{palabra_clave}%"
    connection = mysql.connector.connect(**DB_CONFIG)
    cursor = connection.cursor(dictionary=True)
    query = """
        SELECT u.*, t.id AS transport_id, t.domain, t.transport
        FROM users u
        JOIN transports t ON u.dominio = t.id
        WHERE u.deleted_at IS NULL AND t.deleted_at IS NULL
          AND (u.login LIKE %s OR u.email LIKE %s OR u.maildir LIKE %s 
               OR u.identificacion LIKE %s OR u.grupo LIKE %s OR t.domain LIKE %s);
    """
    cursor.execute(query, (termino, termino, termino, termino, termino, termino))
    resultados = cursor.fetchall()
    cursor.close()
    connection.close()
    return resultados

def buscar_en_aliases_sql(palabra_clave):
    termino = f"%{palabra_clave}%"
    connection = mysql.connector.connect(**DB_CONFIG)
    cursor = connection.cursor(dictionary=True)
    query = """
        SELECT * FROM aliases
        WHERE deleted_at IS NULL
          AND (local LIKE %s OR remoto LIKE %s);
    """
    cursor.execute(query, (termino, termino))
    resultados = cursor.fetchall()
    cursor.close()
    connection.close()
    return resultados

def formatear_usuario(usuario):
    return {
        "ID": usuario["id"],
        "Userid": usuario["userid"],
        "Login": usuario["login"],
        "Email": usuario["email"],
        "Maildir": usuario["maildir"],
        "Identificacion": usuario["identificacion"],
        "Grupo": usuario["grupo"],
        "Dominio": usuario["dominio"],
        "Quota": usuario["quota"],
        "Transport": {
            "ID": usuario["transport_id"],
            "Domain": usuario["domain"],
            "Transport": usuario["transport"]
        }
    }

def obtener_usuario_por_id(user_id):
    return next((u for u in usuarios if u["id"] == user_id), None)

def obtener_alias_por_id(alias_id):
    return next((a for a in aliases if a["id"] == alias_id), None)

@app.middleware("http")
async def check_master_key(request: Request, call_next):
    # solo proteger las rutas de búsqueda
    if request.url.path.startswith("/buscar"):
        token = request.headers.get("X-Master-Key")
        if token != MASTER_KEY:
            raise HTTPException(status_code=403, detail="Forbidden: Invalid MASTER_KEY")
    return await call_next(request)


@app.get("/buscar_user")
def endpoint_buscar_user(query: str, page: int = 1, limit: int = 10, threshold: float = 0.45):
    palabras = query.strip().split()
    data = []

    if len(palabras) == 1:
        resultados = buscar_en_users_sql(query)
        data.extend(resultados)
    else:
        resultados = buscar(index_users, id_map_users, usuarios, query, threshold)
        for user_id, _ in resultados:
            usuario = obtener_usuario_por_id(user_id)
            if usuario:
                data.append(usuario)
    data = [formatear_usuario(u) for u in data]

    total = len(data)
    total_pages = math.ceil(total / limit)
    start, end = (page - 1) * limit, (page - 1) * limit + limit
    return JSONResponse({
        "TotalCount": total,
        "TotalPages": total_pages,
        "Users": data[start:end]
    })

@app.get("/buscar_alias")
def endpoint_buscar_alias(query: str, page: int = 1, limit: int = 10, threshold: float = 0.35):
    palabras = query.strip().split()
    data = []

    if len(palabras) == 1:
        resultados = buscar_en_aliases_sql(query)
        data.extend(resultados)
    else:
        resultados = buscar(index_aliases, id_map_aliases, aliases, query, threshold)
        for alias_id, _ in resultados:
            alias = obtener_alias_por_id(alias_id)
            if alias:
                data.append(alias)
    print(type(data))
    print(data)
    
    total = len(data)
    total_pages = math.ceil(total / limit)
    start, end = (page - 1) * limit, (page - 1) * limit + limit
    
    return JSONResponse(content=jsonable_encoder({
        "TotalCount": total,
        "TotalPages": total_pages,
        "Alias": data[start:end]
    }))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
