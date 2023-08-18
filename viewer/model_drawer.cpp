#include "model_drawer.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

#include <GL/glew.h>

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "tiny_gltf.h"

struct texture_t
{
    GLuint id;
};

struct buffer_t
{
    std::vector<std::uint8_t> data;
    std::size_t elem_size;
    GLuint id;
};

struct primitive_t
{
    std::map<std::string, buffer_t> buffers;
    buffer_t index_buffer;
};

struct node_t
{
    glm::mat4 transform;
    node_t* parent;

    glm::mat4 compute_global_transform() const
    {
        if (parent == nullptr)
        {
            return transform;
        }
        return parent->compute_global_transform() * transform;
    }
};

struct resources_t
{
    std::map<std::size_t, texture_t> textures;
    std::map<const tinygltf::Primitive *, primitive_t> primitives;
    std::map<const tinygltf::Node *, node_t> nodes;
    std::map<const tinygltf::Mesh*, std::vector<float>> weights;
    std::map<std::string, float> blend_weights;
    GLint shader;
    glm::mat4 wvp;
    std::array<glm::mat4, 160> transforms;
};

static int load_shader(GLuint shaderObj, std::string fileName)
{
    std::ifstream ifs(fileName);
    if (!ifs)
    {
        std::cout << "error" << std::endl;
        return -1;
    }

    std::string source;
    std::string line;
    while (getline(ifs, line))
    {
        source += line + "\n";
    }

    const GLchar *sourcePtr = (const GLchar *)source.c_str();
    GLint length = source.length();
    glShaderSource(shaderObj, 1, &sourcePtr, &length);

    return 0;
}

static GLint load_program(std::string vertexFileName, std::string fragmentFileName)
{
    // シェーダーオブジェクト作成
    GLuint vertShaderObj = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragShaderObj = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint shader;

    // シェーダーコンパイルとリンクの結果用変数
    GLint compiled, linked;

    /* シェーダーのソースプログラムの読み込み */
    if (load_shader(vertShaderObj, vertexFileName))
        return -1;
    if (load_shader(fragShaderObj, fragmentFileName))
        return -1;

    /* バーテックスシェーダーのソースプログラムのコンパイル */
    glCompileShader(vertShaderObj);
    glGetShaderiv(vertShaderObj, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_FALSE)
    {
        fprintf(stderr, "Compile error in vertex shader.\n");
        return -1;
    }

    /* フラグメントシェーダーのソースプログラムのコンパイル */
    glCompileShader(fragShaderObj);
    glGetShaderiv(fragShaderObj, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_FALSE)
    {
        fprintf(stderr, "Compile error in fragment shader.\n");
        return -1;
    }

    /* プログラムオブジェクトの作成 */
    shader = glCreateProgram();

    /* シェーダーオブジェクトのシェーダープログラムへの登録 */
    glAttachShader(shader, vertShaderObj);
    glAttachShader(shader, fragShaderObj);

    /* シェーダーオブジェクトの削除 */
    glDeleteShader(vertShaderObj);
    glDeleteShader(fragShaderObj);

    /* シェーダープログラムのリンク */
    glLinkProgram(shader);
    glGetProgramiv(shader, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE)
    {
        fprintf(stderr, "Link error.\n");
        return -1;
    }

    return shader;
}

class model_drawer::model_data
{
public:
    tinygltf::Model model;

    resources_t resources;

    ~model_data() = default;
};

static void flip_axis(float* buffer, std::size_t length, bool x_axis, bool y_axis, bool z_axis)
{
    for (std::size_t i = 0; i < length; i++)
    {
        glm::vec3 v(buffer[i * 3], buffer[i * 3 + 1], buffer[i * 3 + 2]);
        if (x_axis) v.x = -v.x;
        if (y_axis) v.y = -v.y;
        if (z_axis) v.z = -v.z;
        buffer[i * 3] = v.x;
        buffer[i * 3 + 1] = v.y;
        buffer[i * 3 + 2] = v.z;
    }
}

static constexpr std::uint8_t* null_pointer = nullptr;

#define BUFFER_OFFSET(i) (null_pointer + (i))

static std::size_t get_elem_size_from_type(int type)
{
    std::size_t elem_size = 1;
    if (type == TINYGLTF_TYPE_SCALAR)
    {
        elem_size = 1;
    }
    else if (type == TINYGLTF_TYPE_VEC2)
    {
        elem_size = 2;
    }
    else if (type == TINYGLTF_TYPE_VEC3)
    {
        elem_size = 3;
    }
    else if (type == TINYGLTF_TYPE_VEC4)
    {
        elem_size = 4;
    }
    else
    {
        assert(0);
    }
    return elem_size;
}

static void draw_mesh(tinygltf::Model &model, const tinygltf::Mesh &mesh, resources_t& resources)
{
    for (size_t i = 0; i < mesh.primitives.size(); i++)
    {
        const tinygltf::Primitive &primitive = mesh.primitives[i];

        if (primitive.indices < 0)
        {
            continue;
        }

        const tinygltf::Accessor &index_accessor =
            model.accessors[primitive.indices];

        primitive_t* prim;
        if (auto iter = resources.primitives.find(&primitive); iter != resources.primitives.end())
        {
            prim = &iter->second;
        }
        else
        {
            resources.primitives[&primitive] = primitive_t{};
            prim = &resources.primitives[&primitive];
            for (const auto& [attrib_name, attrib_data] : primitive.attributes)
            {
                assert(attrib_data >= 0);
                const tinygltf::Accessor &accessor = model.accessors[attrib_data];

                const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
                std::size_t elem_size = get_elem_size_from_type(accessor.type);
                if ((attrib_name == "POSITION") ||
                    (attrib_name == "NORMAL") ||
                    (attrib_name == "TEXCOORD_0") ||
                    (attrib_name == "JOINTS_0") ||
                    (attrib_name == "WEIGHTS_0"))
                {
                    const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];
                    const auto data_ptr = &buffer.data[bufferView.byteOffset];
                    std::vector<std::uint8_t> base_data(data_ptr, data_ptr + bufferView.byteLength);
                    std::vector<std::uint8_t> data(data_ptr, data_ptr + bufferView.byteLength);

                    for (int target_id = 0; target_id < primitive.targets.size(); target_id++)
                    {
                        const auto& target = primitive.targets[target_id];
                        if (resources.weights.find(&mesh) == resources.weights.end())
                        {
                            continue;
                        }

                        const auto weights = resources.weights.at(&mesh);
                        const auto weight = target_id < weights.size() ? weights.at(target_id) : 0.0;
                        if (const auto found = target.find(attrib_name); found != target.end())
                        {
                            const auto target_attrib_data = found->second;
                            const tinygltf::Accessor& accessor = model.accessors[target_attrib_data];

                            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
                            const auto data_ptr = &buffer.data[bufferView.byteOffset];

                            for (size_t i = 0; i < data.size() / sizeof(float); i++)
                            {
                                ((float*)data.data())[i] += ((float*)data_ptr)[i] * weight;
                            }
                        }
                    }

                    if (attrib_name == "POSITION")
                    {
                        flip_axis((float*)data.data(), data.size() / sizeof(float) / 3, true, false, true);
                    }

                    prim->buffers[attrib_name] = buffer_t{std::move(data), elem_size, 0};

                    buffer_t* buf = &prim->buffers[attrib_name];
                    glGenBuffers(1, &buf->id);
                    glBindBuffer(bufferView.target, buf->id);
                    glBufferData(bufferView.target, buf->data.size(), buf->data.data(), GL_STATIC_DRAW);
                }
            }

            {
                const tinygltf::BufferView &bufferView = model.bufferViews[index_accessor.bufferView];
                const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

                const auto data_ptr = &buffer.data[bufferView.byteOffset];
                std::vector<std::uint8_t> data(data_ptr, data_ptr + bufferView.byteLength);
                prim->index_buffer = buffer_t{std::move(data), 1, 0};

                buffer_t *buf = &prim->index_buffer;
                glGenBuffers(1, &buf->id);
                glBindBuffer(bufferView.target, buf->id);

                glBufferData(bufferView.target, bufferView.byteLength,
                             &buffer.data.at(0) + bufferView.byteOffset,
                             GL_STATIC_DRAW);
            }
        }

        int mode = -1;
        if (primitive.mode == TINYGLTF_MODE_TRIANGLES)
        {
            mode = GL_TRIANGLES;
        }
        else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_STRIP)
        {
            mode = GL_TRIANGLE_STRIP;
        }
        else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_FAN)
        {
            mode = GL_TRIANGLE_FAN;
        }
        else if (primitive.mode == TINYGLTF_MODE_POINTS)
        {
            mode = GL_POINTS;
        }
        else if (primitive.mode == TINYGLTF_MODE_LINE)
        {
            mode = GL_LINES;
        }
        else if (primitive.mode == TINYGLTF_MODE_LINE_LOOP)
        {
            mode = GL_LINE_LOOP;
        }
        else
        {
            assert(0);
        }

        glUseProgram(resources.shader);

        glUniformMatrix4fv(glGetUniformLocation(resources.shader, "pvw"), 1, GL_FALSE, &resources.wvp[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(resources.shader, "transforms"), resources.transforms.size(), GL_FALSE, &resources.transforms[0][0][0]);
        glUniform1i(glGetUniformLocation(resources.shader, "tex"), 0);
        if (primitive.material >= 0)
        {
            tinygltf::Material &mat = model.materials[primitive.material];
            if (mat.values.find("baseColorTexture") != mat.values.end())
            {
                int texture_index = mat.values["baseColorTexture"].TextureIndex();
                GLuint texture = resources.textures.at(texture_index).id;
                glBindTexture(GL_TEXTURE_2D, texture);
            }
            if (mat.values.find("baseColorFactor") != mat.values.end())
            {
                const auto color = mat.values["baseColorFactor"].ColorFactor();
                float values[] = {(float)color[0], (float)color[1], (float)color[2], (float)color[3]};
                // glMaterialfv(GL_FRONT, GL_DIFFUSE, values);
                glUniform4fv(glGetUniformLocation(resources.shader, "diffuse"), 1, values);
            }
        }

        for (auto iter = prim->buffers.begin(); iter != prim->buffers.end(); iter++)
        {
            const tinygltf::Accessor &accessor = model.accessors[primitive.attributes.at(iter->first)];
            const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];

            if (iter->first == "POSITION")
            {
                const buffer_t *buffer = &iter->second;
                glBindBuffer(GL_ARRAY_BUFFER, buffer->id);
                int byteStride =
                    accessor.ByteStride(model.bufferViews[accessor.bufferView]);
                assert(byteStride != -1);
                glVertexAttribPointer(0, buffer->elem_size,
                                      accessor.componentType,
                                      accessor.normalized ? GL_TRUE : GL_FALSE,
                                      byteStride, BUFFER_OFFSET(accessor.byteOffset));
            }
            else if(iter->first == "NORMAL")
            {
                const buffer_t *buffer = &iter->second;
                glBindBuffer(GL_ARRAY_BUFFER, buffer->id);
                int byteStride =
                    accessor.ByteStride(model.bufferViews[accessor.bufferView]);
                assert(byteStride != -1);
                glVertexAttribPointer(1, buffer->elem_size,
                                      accessor.componentType,
                                      accessor.normalized ? GL_TRUE : GL_FALSE,
                                      byteStride, BUFFER_OFFSET(accessor.byteOffset));
            }
            else if (iter->first == "TEXCOORD_0")
            {
                const buffer_t *buffer = &iter->second;
                glBindBuffer(GL_ARRAY_BUFFER, buffer->id);
                int byteStride =
                    accessor.ByteStride(model.bufferViews[accessor.bufferView]);
                assert(byteStride != -1);
                glVertexAttribPointer(2, buffer->elem_size,
                                      accessor.componentType,
                                      accessor.normalized ? GL_TRUE : GL_FALSE,
                                      byteStride, BUFFER_OFFSET(accessor.byteOffset));
            }
            else if (iter->first == "JOINTS_0")
            {
                const buffer_t *buffer = &iter->second;
                glBindBuffer(GL_ARRAY_BUFFER, buffer->id);
                int byteStride =
                    accessor.ByteStride(model.bufferViews[accessor.bufferView]);
                assert(byteStride != -1);
                glVertexAttribIPointer(3, buffer->elem_size,
                                       accessor.componentType,
                                       byteStride, BUFFER_OFFSET(accessor.byteOffset));
            }
            else if (iter->first == "WEIGHTS_0")
            {
                const buffer_t *buffer = &iter->second;
                glBindBuffer(GL_ARRAY_BUFFER, buffer->id);
                int byteStride =
                    accessor.ByteStride(model.bufferViews[accessor.bufferView]);
                assert(byteStride != -1);
                glVertexAttribPointer(4, buffer->elem_size,
                                      accessor.componentType,
                                      accessor.normalized ? GL_TRUE : GL_FALSE,
                                      byteStride, BUFFER_OFFSET(accessor.byteOffset));
            }
        }
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        glEnableVertexAttribArray(3);
        glEnableVertexAttribArray(4);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prim->index_buffer.id);
        glDrawElements(mode, index_accessor.count, index_accessor.componentType,
                       BUFFER_OFFSET(index_accessor.byteOffset));

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);
        glDisableVertexAttribArray(3);
        glDisableVertexAttribArray(4);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);
    }
}

static void draw_node(tinygltf::Model &model, const tinygltf::Node &node, resources_t &resources)
{
    if (node.mesh > -1)
    {
        assert(node.mesh < static_cast<int>(model.meshes.size()));

        for (auto &transform : resources.transforms)
        {
            transform = glm::mat4(1.f);
        }

        const node_t &mesh_node = resources.nodes.at(&node);

        if (node.skin >= 0)
        {
            const tinygltf::Skin &skin = model.skins[node.skin];
            const tinygltf::Accessor& inv_bind_accessor =
                model.accessors[skin.inverseBindMatrices];

            const tinygltf::BufferView& inv_bind_buffer_view = model.bufferViews[inv_bind_accessor.bufferView];
            const tinygltf::Buffer& inv_bind_buffer = model.buffers[inv_bind_buffer_view.buffer];

            const auto data_ptr = reinterpret_cast<const float*>(&inv_bind_buffer.data[inv_bind_buffer_view.byteOffset + inv_bind_accessor.byteOffset]);

            for (std::size_t j = 0; j < skin.joints.size(); j++)
            {
                std::size_t joint = skin.joints[j];
                const node_t& joint_node = resources.nodes.at(&model.nodes[joint]);
                auto bind = glm::inverse(glm::mat4(glm::make_mat4(data_ptr + j * 16)));
                const auto bind_rot = glm::quat_cast(bind);
                const auto bind_trans = glm::vec3(bind[3]);
                bind = glm::translate(glm::vec3(-bind_trans.x, bind_trans.y, -bind_trans.z)) * glm::toMat4(glm::quat(bind_rot.w, bind_rot.x, -bind_rot.y, bind_rot.z));
                const auto inv_bind = glm::inverse(bind);
                // std::cout << model.nodes[joint].name << ":" << glm::to_string(inv_bind) << std::endl;;
                //resources.transforms[j] = glm::inverse(mesh_node.transform) * joint_node.compute_global_transform() * inv_bind;
                resources.transforms[j] = joint_node.compute_global_transform() * inv_bind;
            }
        }

        draw_mesh(model, model.meshes[node.mesh], resources);
    }

    for (size_t i = 0; i < node.children.size(); i++)
    {
        assert(node.children[i] < static_cast<int>(model.nodes.size()));
        draw_node(model, model.nodes[node.children[i]], resources);
    }
}

static void update_node(tinygltf::Model &model, const tinygltf::Node &node, resources_t &resources)
{
    glm::mat4 transform(1.0f);
    if (node.matrix.size() == 16)
    {
        // Use `matrix' attribute
        transform = glm::mat4(glm::make_mat4(node.matrix.data()));
    }
    else
    {
        // Assume Trans x Rotate x Scale order
        if (node.scale.size() == 3)
        {
            transform = glm::scale(glm::vec3((float)node.scale[0], (float)node.scale[1], (float)node.scale[2])) * transform;
        }

        if (node.rotation.size() == 4)
        {
            transform = glm::toMat4(glm::quat(
                static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]), static_cast<float>(-node.rotation[1]), static_cast<float>(node.rotation[2]))) * transform;
        }

        if (node.translation.size() == 3)
        {
            transform = glm::translate(glm::vec3((float)-node.translation[0], (float)node.translation[1], (float)-node.translation[2])) * transform;
        }
    }
    
    node_t* _node = &resources.nodes.at(&node);
    _node->transform = transform;

    for (size_t i = 0; i < node.children.size(); i++)
    {
        assert(node.children[i] < static_cast<int>(model.nodes.size()));

        const tinygltf::Node &child = model.nodes[node.children[i]];
        update_node(model, child, resources);
    }
}

static void apply_blend_shape(tinygltf::Model& model, resources_t& resources)
{
    const auto& blendshape = model.extensions.at("VRM").Get("blendShapeMaster");
    assert(blendshape.IsObject());

    const auto& groups = blendshape.Get("blendShapeGroups");
    assert(groups.IsArray());

    for (int i = 0; i < groups.Size(); i++)
    {
        const auto& group = groups.Get(i);

        const auto& is_binary = group.Get("isBinary").Get<bool>();
        const auto& name = group.Get("name").Get<std::string>();
        const auto& preset_name = group.Get("presetName").Get<std::string>();
        const auto& binds = group.Get("binds");

        const auto blend_weight = resources.blend_weights.find(name) != resources.blend_weights.end() ? resources.blend_weights.at(name) : 0.0;

        for (int j = 0; j < binds.Size(); j++)
        {
            const auto& bind = binds.Get(j);

            const auto mesh = bind.Get("mesh").Get<int>();
            const auto index = bind.Get("index").Get<int>();
            const auto weight = bind.Get("weight").Get<double>();

            auto& weights = resources.weights[&model.meshes[mesh]];
            weights.resize(index + 1);
            weights[index] = weight / 100 * blend_weight;
        }
    }
}

static void draw_model(tinygltf::Model &model, resources_t &resources)
{
    assert(model.scenes.size() > 0);
    int scene_to_display = model.defaultScene > -1 ? model.defaultScene : 0;
    const tinygltf::Scene &scene = model.scenes[scene_to_display];

    apply_blend_shape(model, resources);

    for (size_t i = 0; i < scene.nodes.size(); i++)
    {
        draw_node(model, model.nodes[scene.nodes[i]], resources);
    }
}

static void update_model(tinygltf::Model &model, resources_t &resources)
{
    for (size_t i = 0; i < model.nodes.size(); i++)
    {
        update_node(model, model.nodes[i], resources);
    }
}

model_drawer::model_drawer(std::string path)
    : data(new model_data())
{
    std::string err;
    std::string warn;

    tinygltf::TinyGLTF loader;
    bool ret = loader.LoadBinaryFromFile(&data->model, &err, &warn, path);

    if (!warn.empty())
    {
        printf("Warn: %s\n", warn.c_str());
    }

    if (!err.empty())
    {
        printf("Err: %s\n", err.c_str());
    }

    if (!ret)
    {
        printf("Failed to parse glTF\n");
        return;
    }

    for (std::size_t i = 0; i < data->model.textures.size(); i++)
    {
        const auto &tex = data->model.textures[i];
        {
            tinygltf::Image &image = data->model.images[tex.source];
            GLuint texId;
            int target = GL_TEXTURE_2D;
            glGenTextures(1, &texId);
            glBindTexture(target, texId);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

            // Ignore Texture.fomat.
            GLenum format = GL_RGBA;
            if (image.component == 3)
            {
                format = GL_RGB;
            }
            glTexImage2D(target, 0, format, image.width,
                         image.height, 0, format, GL_UNSIGNED_BYTE,
                         &image.image.at(0));

            // glGenerateMipmap(GL_TEXTURE_2D);
            glBindTexture(target, 0);

            data->resources.textures[i] = texture_t{texId};
        }
    }

    data->resources.shader = load_program("../viewer/shaders/shader.vert", "../viewer/shaders/shader.frag");

    for (std::size_t i = 0; i < data->model.nodes.size(); i++)
    {
        const auto &node = data->model.nodes[i];
        data->resources.nodes[&node] = node_t{glm::mat4(1.f), nullptr};
    }

    for (std::size_t i = 0; i < data->model.nodes.size(); i++)
    {
        const auto &node = data->model.nodes[i];
        for (const auto j : node.children)
        {
            const auto &child = data->model.nodes[j];
            data->resources.nodes.at(&child).parent = &data->resources.nodes.at(&node);
        }
    }
    if (data->model.extensions.find("VRM") != data->model.extensions.end())
    {
        const auto& vrm = data->model.extensions.at("VRM");
        const auto& humanoid = vrm.Get("humanoid");
        const auto& bones = humanoid.Get("humanBones");

        const auto to_double_vector = [](const tinygltf::Value& value)
        {
            std::vector<double> values;

            if (value.IsArray())
            {
                for (std::size_t i = 0; i < value.Size(); i++)
                {
                    values.push_back(value.Get(i).Get<double>());
                }
            }
            return values;
        };
        for (std::size_t i = 0; i < bones.Size(); i++)
        {
            const auto& bone = bones.Get(i);

            humanoid_bone _bone;
            _bone.name = bone.Get("bone").Get<std::string>();
            _bone.node = bone.Get("node").Get<int>();
            _bone.use_default_values = bone.Get("useDefaultValues").Get<bool>();
            _bone.min_values = to_double_vector(bone.Get("min"));
            _bone.max_values = to_double_vector(bone.Get("max"));
            _bone.center_values = to_double_vector(bone.Get("center"));
            _bone.axis_length = bone.Get("axisLength").Get<double>();
            bone_map.insert(std::make_pair(_bone.name, _bone));
        }
    }

    update_model(data->model, data->resources);
}

model_drawer::~model_drawer() = default;

void model_drawer::draw(glm::mat4 wvp)
{
    data->resources.wvp = wvp;
    draw_model(data->model, data->resources);
}

void model_drawer::set_bone_transform(const std::string &name, const glm::mat4 &value)
{
    const auto &bone = bone_map.at(name);
    const auto &node = data->model.nodes[bone.node];
    node_t &_node = data->resources.nodes.at(&node);
    _node.transform = value;
}
glm::mat4 model_drawer::get_bone_transform(const std::string &name) const
{
    const auto &bone = bone_map.at(name);
    const auto &node = data->model.nodes[bone.node];
    node_t &_node = data->resources.nodes.at(&node);
    return _node.transform;
}
glm::mat4 model_drawer::get_bone_global_transform(const std::string &name) const
{
    const auto &bone = bone_map.at(name);
    const auto &node = data->model.nodes[bone.node];
    node_t &_node = data->resources.nodes.at(&node);
    return _node.compute_global_transform();
}

std::map<std::string, glm::mat4> model_drawer::get_bone_transforms() const
{
    std::map<std::string, glm::mat4> result;
    for (const auto& [name, bone] : bone_map)
    {
        result.insert(std::make_pair(name, get_bone_transform(name)));
    }
    return result;
}
void model_drawer::set_bone_transforms(const std::map<std::string, glm::mat4>& values)
{
    for (const auto& [name, value] : values)
    {
        set_bone_transform(name, value);
    }
}

void model_drawer::set_blend_weight(std::string name, float value)
{
    data->resources.blend_weights[name] = value;
}
float model_drawer::get_blend_weight(std::string name) const
{
    return data->resources.blend_weights.find(name) != data->resources.blend_weights.end() ? data->resources.blend_weights.at(name) : 0.0f;
}
