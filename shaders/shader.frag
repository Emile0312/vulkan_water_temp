#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragPosition;
layout(location = 2) in vec2 screenUv;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 camPosition;
    vec3 lightsource;
} ubo;

layout(binding = 1) uniform OptionBufferObject {
    float l0; // 0.04
    float k; //1.0
    float a; //0.05
    float g; // gravité !!
    float delta_t; //variable 
    float max_dist; //1.0 - 1.1
    uint nb_particles; // 300000000000 !
    uint nb_triangles;
    uint nb_points;
    uint nb_chunks;
    float color; // ~ 3
    float epsilon; // 0.001
    float blending;
} opt;

void main() {
    outColor = vec4(fragPosition,1);

}
