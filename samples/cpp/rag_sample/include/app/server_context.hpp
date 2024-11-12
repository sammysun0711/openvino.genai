#pragma once

#include "util.hpp"

void handle_db_init(util::ServerContext& server_context_ref, bool setup = false);

void handle_image_embeddings_init(util::ServerContext& server_context_ref);

std::vector<std::string> handle_db_retrieval_image(util::ServerContext& server_context_ref,
                                                   std::vector<std::string> image_paths, int topk = 3, bool debug = false);

bool is_image_embeddings_ready(util::ServerContext& server_context_ref);