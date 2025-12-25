#define _USE_MATH_DEFINES
#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif

#include "fbx_loader.hpp"

#include <ofbx.h>

#include <cstring>
#include <fstream>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <iostream>
#include <vector>

namespace marionette::preprocess {
static glm::mat4 ofbx_matrix_to_glm(const ofbx::DMatrix &m) {
  // OpenFBX returns column-major matrix (same as GLM)
  return glm::mat4(
      static_cast<float>(m.m[0]), static_cast<float>(m.m[1]), static_cast<float>(m.m[2]),
      static_cast<float>(m.m[3]), static_cast<float>(m.m[4]), static_cast<float>(m.m[5]),
      static_cast<float>(m.m[6]), static_cast<float>(m.m[7]), static_cast<float>(m.m[8]),
      static_cast<float>(m.m[9]), static_cast<float>(m.m[10]), static_cast<float>(m.m[11]),
      static_cast<float>(m.m[12]), static_cast<float>(m.m[13]), static_cast<float>(m.m[14]),
      static_cast<float>(m.m[15]));
}

[[maybe_unused]] static glm::vec3 ofbx_vec3_to_glm(const ofbx::Vec3 &v) {
  return glm::vec3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}

// Recursively process FBX object hierarchy
static void process_object(const ofbx::Object *obj, const std::shared_ptr<node_t> &parent_node) {
  if (!obj) return;

  std::shared_ptr<node_t> current_node = std::make_shared<node_t>();
  current_node->name = obj->name;
  current_node->transform = ofbx_matrix_to_glm(obj->getLocalTransform());
  current_node->parent = parent_node;
  parent_node->children.push_back(current_node);

  // Check if this object has geometry (mesh) attributes
  if (obj->getType() == ofbx::Object::Type::MESH) {
    const ofbx::Mesh *mesh = static_cast<const ofbx::Mesh *>(obj);
    auto mesh_node = std::make_shared<mesh_node_t>();
    mesh_node->name = mesh->name;
    mesh_node->transform = glm::mat4(1.0f);

    // Extract skin weights
    const ofbx::Skin *skin = mesh->getSkin();
    if (skin) {
      int cluster_count = skin->getClusterCount();
      for (int i = 0; i < cluster_count; i++) {
        const ofbx::Cluster *cluster = skin->getCluster(i);
        const ofbx::Object *link = cluster->getLink();
        if (link && cluster->getIndicesCount() > 0) {
          const double *weights = cluster->getWeights();
          mesh_node->weights.insert(std::make_pair(link->name, static_cast<float>(weights[0])));
        }
      }
    }

    mesh_node->parent = current_node;
    current_node->children.push_back(mesh_node);
  }
  // Check if this is a skeleton/bone node
  else if (obj->getType() == ofbx::Object::Type::LIMB_NODE) {
    auto skeleton_node = std::make_shared<skeleton_node_t>();
    skeleton_node->name = obj->name;
    skeleton_node->transform = glm::mat4(1.0f);
    skeleton_node->parent = current_node;
    current_node->children.push_back(skeleton_node);
  }

  // Process all child objects
  int child_idx = 0;
  while (const ofbx::Object *child = obj->resolveObjectLink(child_idx++)) {
    if (child && child->isNode()) {
      process_object(child, current_node);
    }
  }
}

std::shared_ptr<node_t> load_model(const std::string &file_name) {
  // Load FBX file
  std::ifstream file(file_name, std::ios::binary);
  if (!file.is_open()) {
    std::cout << "Failed to open the FBX file: " << file_name << std::endl;
    exit(-1);
  }

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<ofbx::u8> content(size);
  file.read(reinterpret_cast<char *>(content.data()), size);
  file.close();

  // Parse FBX (we need geometry for meshes and bones for skeletons)
  ofbx::IScene *scene =
      ofbx::load(content.data(), size, static_cast<ofbx::u16>(ofbx::LoadFlags::NONE));
  if (!scene) {
    std::cout << "Failed to parse FBX file: " << ofbx::getError() << std::endl;
    exit(-1);
  }

  // Get the root object of the scene
  const ofbx::Object *fbx_root = scene->getRoot();
  if (!fbx_root) {
    std::cout << "FBX file has no root object" << std::endl;
    scene->destroy();
    exit(-1);
  }

  // Create root node
  auto root = std::make_shared<node_t>();
  root->name = fbx_root->name;
  root->transform = ofbx_matrix_to_glm(fbx_root->getLocalTransform());

  // Process all children of root
  int child_idx = 0;
  while (const ofbx::Object *child = fbx_root->resolveObjectLink(child_idx++)) {
    if (child && child->isNode()) {
      process_object(child, root);
    }
  }

  scene->destroy();

  // Return root node with all children
  return root;
}
}  // namespace marionette::preprocess
