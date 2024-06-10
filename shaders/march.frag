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

struct Particle {
    vec3 position;
    vec3 velocity;
};

struct Statend {
    uint start;
    uint end;
};



layout(std140, binding = 2) readonly buffer ParticleSSBOIn {
    Particle particlesIn[ ];
};

layout(std430, binding = 6) buffer ChunkSSBO {
    uint chunks[ ];
};
layout(std430, binding = 7) buffer IndexSSBOIn {
    uint particlesIndex[ ];
};
layout(std430, binding = 8) buffer ChunkIndexSSBOIn {
    Statend chunkIndex[ ];
};

layout(binding = 14) uniform sampler2D texSampler;


float smin( float a, float b, float k )
{
    k *= 1.0;
    float r = exp2(-a/k) + exp2(-b/k);
    return -k*log2(r);
}


uint hash(ivec3 pos)
{
    return(uint(mod(pos.x*727 + pos.y*359 + pos.z*547, opt.nb_chunks)));
}


ivec3 to_chunk_coords(vec3 point)
{
    const float size = opt.l0*opt.max_dist;
    return(ivec3(  int( (point.x )/size),
                    int( (point.y )/size), 
                    int( (point.z )/size)
                    ));
}



float distance_to_objects(vec3 point)
{
    const ivec3 voisins[27] = {
        ivec3(-1,-1,-1),
        ivec3(-1,-1,0),
        ivec3(-1,-1,1),
        ivec3(0,-1,-1),
        ivec3(0,-1,0),
        ivec3(0,-1,1),
        ivec3(1,-1,-1),
        ivec3(1,-1,0),
        ivec3(1,-1,1),

        ivec3(-1,0,-1),
        ivec3(-1,0,0),
        ivec3(-1,0,1),
        ivec3(0,0,-1),
        ivec3(0,0,0),
        ivec3(0,0,1),
        ivec3(1,0,-1),
        ivec3(1,0,0),
        ivec3(1,0,1),

        ivec3(-1,1,-1),
        ivec3(-1,1,0),
        ivec3(-1,1,1),
        ivec3(0,1,-1),
        ivec3(0,1,0),
        ivec3(0,1,1),
        ivec3(1,1,-1),
        ivec3(1,1,0),
        ivec3(1,1,1),
    };
    ivec3 chunk_coords = to_chunk_coords(point);
    float dnew = min(distance(point,particlesIn[0].position) - opt.l0/3, opt.l0*opt.max_dist);
    for(int i = 0; i < 27; i++)
    {
        uint chunk_pos = hash(chunk_coords + voisins[i]);
        for(uint j = chunkIndex[chunk_pos].start; j < chunkIndex[chunk_pos].end; j++)
        {
            float d1 = distance(point, particlesIn[particlesIndex[j]].position) - opt.l0/3;
            dnew = smin(d1,dnew,opt.blending);
        }
    }
    return(dnew);
}

vec3 findNormal(vec3 position)
{
    float distance = distance_to_objects(position);
    vec2 epsilon = vec2(opt.epsilon,0);
    vec3 n = distance - vec3(
        distance_to_objects(position-epsilon.xyy), //X Component
        distance_to_objects(position-epsilon.yxy), //Y Component
        distance_to_objects(position-epsilon.yyx)  //Z Component
    );

    return normalize(n);
}

int nearest_particle(vec3 point, inout float dnew)
{
    int pmin = -1;
    for(int i = 0; i < opt.nb_particles; i++)
    {
        float d = abs(distance(point, particlesIn[i].position) - opt.l0/3);
        if (d <= 0.0)
        {
            dnew = 0.0;
            return(i);
        }
        if(d < dnew) 
        {
            pmin = i;
            dnew = d;
        }
    }
    return(pmin);
}

//le code de qqun d'autre !!! parce que j'y comprend rien.
// All components are in the range [0…1], including hue.
vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {

    vec3 dx = dFdx(fragPosition);
    vec3 dy = dFdy(fragPosition);
    vec3 N = normalize(cross(dx, dy));
    /*float magnitude = min (1/dot(ubo.lightsource - fragPosition,ubo.lightsource - fragPosition), 1.0);
    vec3 reflection = normalize ((ubo.lightsource - fragPosition) - 2*dot(ubo.lightsource - fragPosition, N)*N);
    float reflection_part = 0.0 + abs(dot(reflection, normalize( CamPosition - fragPosition)))*1.0; */
    float dim = abs(dot(N, ubo.lightsource));
    vec3 surface_color = 0.8*dim*fragColor;
    float dmax = distance(ubo.camPosition, fragPosition);
    vec3 direction = normalize(fragPosition - ubo.camPosition);
    vec3 point = texture(texSampler, screenUv).xyz;
    bool condition = true;
    float epsilon = opt.epsilon;
    outColor = vec4( surface_color, 1.0);
    while(condition)
    {

        float dnew = distance_to_objects(point);

        if(dnew >= dmax)
        {   
            condition = false;
            outColor = vec4( surface_color, 1.0);
        }
        if( dnew < epsilon )
        {
            condition = false;
            //float speed = sqrt(dot(particlesIn[i].velocity,particlesIn[i].velocity));
            vec3 Nsphere = findNormal(point);
            float t = 0.1 + 0.8*(1 - abs(dot(Nsphere,direction)));//transparence
            float dimsphere = abs(dot(N,Nsphere));
            vec3 couleur = (1-t)*surface_color + t*vec3(0,0,1);
            //hsv2rgb(vec3(abs(245 -  opt.color*speed), 1.0,0.8)) 
            outColor = vec4( couleur ,1.0);
        }
        else
        {
            point = point + dnew*direction;
            dmax = dmax - dnew;
            epsilon += 0.002 *dnew;
        }
    }
}
