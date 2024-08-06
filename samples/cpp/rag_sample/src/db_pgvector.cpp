// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "db_pgvector.hpp"

#include <pqxx/except>
#include <pqxx/transaction>
#include <string>

void DBPgvector::db_setup(const std::string& db_connection_str) {
    if (!pgconn) {
        pgconn.emplace(db_connection_str);
        std::cout << "connected to DB: " << pgconn->dbname() << std::endl;
    }
    try {
        pqxx::work tx{*pgconn};
        tx.exec0("CREATE EXTENSION IF NOT EXISTS vector");
        tx.exec0("DROP TABLE IF EXISTS documents");
        // max vector dim is 16000
        // TODO: suppport more embedding model, 
        // 512 is only for BGE-small, 1024 is for BGE-large
        tx.exec0("CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(512))");
        tx.commit();
        std::cout << "Create table document in PostgreSQL" << std::endl;

        } catch (pqxx::sql_error const& e) {
            std::cerr << "SQL error: " << e.what() << '\n';
            }        
}

void DBPgvector::db_store_embeddings(std::vector<std::string> chunks, std::vector<std::vector<float>> embeddings) {
    std::cout << "db_store_embeddings start" << std::endl;
    try {
        pqxx::work tx{*pgconn};

        std::cout << "get chunks and embeddings successed\n";
        std::cout << "document_data: " << chunks.size() << std::endl;
        std::cout << "embeddings_data: " << embeddings.size() << ", " << embeddings[0].size() << std::endl;

        for (size_t i = 0; i < chunks.size(); ++i) {
            // std::cout << "id: " << i << std::endl;
            // if chunks size is too large, try stream to accelerate.
            std::string insert_sql = "INSERT INTO documents (content, embedding) VALUES ($1, $2)";
            // or HalfVector
            pgvector::Vector embeddings_vector(embeddings[i]);

            tx.exec_params(insert_sql, chunks[i], embeddings_vector);
        }
        tx.commit();

        } catch (pqxx::sql_error const& e) {
            std::cerr << "SQL error: " << e.what() << '\n';
        }
    std::cout << "db_store_embeddings successed\n";
}

std::vector<std::string> DBPgvector::db_retrieval(size_t chunk_num,
                                                  std::vector<std::string> query,
                                                  std::vector<std::vector<float>> query_embedding) {
    std::cout << "db_retrieval start and will get the top 3 results: " << std::endl;
    std::vector<std::string> retrieval_res;
    try {
        pqxx::work tx{*pgconn};
        pgvector::Vector embeddings_vector(query_embedding[0]);

        std::string insert_sql = "INSERT INTO documents (content, embedding) VALUES ($1, $2)";

        tx.exec_params(insert_sql, query[0], embeddings_vector);

        // the top 3 search, result without the query
        pqxx::result res{tx.exec_params(
            "SELECT id, content, embedding FROM documents WHERE id != $1 ORDER BY embedding <=> $2 LIMIT 3",
            chunk_num + 1,
            embeddings_vector)};

        // assert(res.size() == 3);

        std::cout << "===================================\n";

        for (const pqxx::row& row : res) {
            std::string chunks_str = row["content"].c_str();
            std::cout << "ID: " << row[0] << ", parts of the chunks content: \n"
                      << chunks_str.substr(0, 100) << " ..." << std::endl;
            std::cout << "===================================\n";
            retrieval_res.push_back(row["content"].c_str());
        }
        tx.commit();
        std::cout << "DBPgvector::db_retrieval SELECT successed\n";

        // TODO:Delete vector for reset
        // tx.exec_params("DELETE id, content, embedding FROM documents WHERE id = $1", chunk_num + 1);

    } catch (pqxx::sql_error const& e) {
        std::cerr << "SQL error: " << e.what() << '\n';
    }
    std::cout << "DBPgvector::db_retrieval DELETE successed\n";
    return retrieval_res;
}