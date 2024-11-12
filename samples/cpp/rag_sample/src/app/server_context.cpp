#include "server_context.hpp"

void handle_db_init(util::ServerContext& server_context_ref, bool setup) {
    if (server_context_ref.db_state == State::STOPPED || server_context_ref.db_state == State::ERR) {
        server_context_ref.db_pgvector_pointer = std::make_shared<DBPgvector>();
        if (setup) {
            server_context_ref.db_pgvector_pointer->db_setup(server_context_ref.args.db_connection);
        } else {
            server_context_ref.db_pgvector_pointer->db_connect(server_context_ref.args.db_connection);
        }
        server_context_ref.db_state = State::IDLE;
    }
}

void handle_image_embeddings_init(util::ServerContext& server_context_ref) {
    if (server_context_ref.image_embeddings_state == State::STOPPED ||
        server_context_ref.image_embeddings_state == State::ERR) {
        server_context_ref.image_embeddings_pointer = std::make_shared<BlipModel>();
        server_context_ref.image_embeddings_pointer->init(server_context_ref.args.image_embedding_model_path,
                                                          server_context_ref.args.image_embedding_device);
        server_context_ref.image_embeddings_state = State::IDLE;
    }
}

std::vector<std::string> handle_db_retrieval_image(util::ServerContext& server_context_ref,
                                                   std::vector<std::string> image_paths,
                                                   int topk,
                                                   bool debug) {
    if (server_context_ref.db_state == State::IDLE) {
        server_context_ref.db_state = State::RUNNING;
        std::vector<std::vector<float>> embeddings_query =
            server_context_ref.image_embeddings_pointer->encode_images(image_paths);
        server_context_ref.retrieval_res =
            server_context_ref.db_pgvector_pointer->db_retrieval_only(image_paths, embeddings_query, topk, debug);
        server_context_ref.db_state = State::IDLE;
        std::cout << "HandleMaster::db_retrieval successed\n";
        return server_context_ref.retrieval_res;
    }
}

bool is_image_embeddings_ready(util::ServerContext& server_context_ref) {
    // TODO: potential data race, but we only read in UI thread, modified in worker thread, it's safe by now
    // but if we want to start a server thread which shares the server context, we should add a mutex inside server
    // context in short: server context is not thread safe
    return server_context_ref.db_state == State::IDLE && server_context_ref.image_embeddings_state == State::IDLE;
}