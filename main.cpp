
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <stdio.h>
#include <vector>
#include <array>
#include <stdbool.h>
#include <cstring>
#include <optional>
#include <set>
#include <cstdint> // uint32_t
#include <limits> // std::numeric_limits
#include <algorithm> // std::clamp
#define GLM_SWIZZLE_XYZW 
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#define VOLK_IMPLEMENTATION

#include <chrono>

const uint32_t WIDTH = 3600;
const uint32_t HEIGHT = 1800;

const int MAX_FRAMES_IN_FLIGHT = 4;

const uint32_t COMPUTE_STEP = 128;
const float SORT_STEP = 256.0;
const float ASSIGN_STEP = 256.0;
const uint32_t STATEND_STEP = 128;

const uint32_t NB_TIMEPOINTS = 60;

const bool JUST_RAY_MARCHING = false;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
// use pvkGetPhysicalDeviceFeatures2KHR

struct Vertex {
    glm::vec3 pos;
    alignas(16) glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);
        return attributeDescriptions;
    }
};

struct SortBufferObject{
    uint32_t begin_swaps;
    uint32_t nb_swaps;
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    alignas(16) glm::vec3 cam;
    alignas(16) glm::vec3 lightsource;
};

struct OptionBufferObject {
    float l0; // 0.04
    float k; //1.0
    float a; //0.05
    float g;
    float delta_t; //variable 
    float max_dist; //1.0 - 1.1
    uint32_t nb_particles; // 300000000000 !
    uint32_t nb_triangles; //yay !
    uint32_t nb_points;
    uint32_t nb_chunks;
    float color; // ~ 3
    float epsilon; // 0.001
    float blending; //0.2
};

struct Particle {
    glm::vec3 position;
    alignas(16) glm::vec3 velocity;
};

struct Statend{
    uint32_t start;
    uint32_t end;
};

struct Swap {
    uint32_t a;
    uint32_t b;
};

const std::vector<Vertex> vertices = {
    {{-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}},
    {{1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}},
    {{-1.0f, 1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}},

    {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
    {{-1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
    {{1.0f, -1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
    {{-1.0f, -1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},

    {{-10.0f, -10.0f, -10.0f}, {0.9f, 0.9f, 0.9f}},
    {{10.0f, -10.0f, -10.0f}, {0.9f, 0.9f, 0.9f}},
    {{10.0f, 10.0f, -10.0f}, {0.9f, 0.9f, 0.9f}},
    {{-10.0f, 10.0f, -10.0f}, {0.9f, 0.9f, 0.9f}},

    {{10.0f, 10.0f, 10.0f}, {0.9f, 0.9f, 0.9f}},
    {{-10.0f, 10.0f, 10.0f}, {0.9f, 0.9f, 0.9f}},
    {{10.0f, -10.0f, 10.0f}, {0.9f, 0.9f, 0.9f}},
    {{-10.0f, -10.0f, 10.0f}, {0.9f, 0.9f, 0.9f}}
    };

const std::vector<uint32_t> indices = {
    1,0,2,
    2,0,3,

    /*6,1,2,
    6,2,4,*/

    3,0,5,
    5,0,7,

    0,1,6,
    0,6,7,

    //couvercle
    5,4,2,
    2,3,5,

    7,6,4,
    7,4,5,

    //second cube !!!
    9,8,10,
    10,8,11,

    14,9,10,
    14,10,12,

    11,8,13,
    13,8,15,

    8,9,14,
    8,14,15,

    //couvercle
    13,12,10,
    10,11,13,

    15,14,12,
    15,12,13,
};

std::vector<Particle> perles = {
    {{0.0f, 0.0f, -0.5f}, {0.0f, 0.0f, 0.0f}},
    {{0.0f, 0.0f, 0.5f}, {0.0f, 0.0f, 0.0f}},
};


struct QueueFamilyIndices {
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> graphicsAndComputeFamily;

    bool isComplete() {
        return presentFamily.has_value() && graphicsAndComputeFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};


class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:

    VkInstance instance;
    
    GLFWwindow* window;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    VkDevice device;

    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    std::vector<Vertex> end_vertices;
    std::vector<uint32_t> end_indices;

    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkQueue computeQueue;

    VkSurfaceKHR surface;

    VkSwapchainKHR swapChain;

    std::vector<VkImage> swapChainImages;

    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    std::vector<VkImageView> swapChainImageViews;


    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSetLayout sortingDescriptorSetLayout;

    VkPipelineLayout pipelineLayout;    
    VkPipeline graphicsPipeline;

    VkPipeline computePipeline;
    VkPipelineLayout computePipelineLayout;

    VkPipeline raymarchingPipeline;
    VkPipelineLayout raymarchingPipelineLayout;


    VkPipeline assignPipeline;
    VkPipelineLayout assignPipelineLayout;

    VkPipeline sortPipeline;
    VkPipelineLayout sortPipelineLayout;

    VkPipeline statendPipeline;
    VkPipelineLayout statendPipelineLayout;

    std::vector<VkFramebuffer> swapChainFramebuffers;
    std::vector<VkFramebuffer> raymarchingFramebuffers;

    VkCommandPool commandPool;

    std::vector<VkImage> raymarchingImages;
    std::vector<VkImageView> raymarchingImageViews;
    std::vector<VkDeviceMemory> raymarchingImagesMemory;
    VkSampler textureSampler;

    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkCommandBuffer> raymarchingCommandBuffers;
    std::vector<VkCommandBuffer> computeCommandBuffers;
    std::vector<VkCommandBuffer> assignCommandBuffers;
    std::vector<VkCommandBuffer> sortCommandBuffers;
    std::vector<VkCommandBuffer> statendCommandBuffers;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    VkDescriptorPool sortingDescriptorPool;
    std::vector<VkDescriptorSet> sortingDescriptorSets;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    std::vector<VkDescriptorSet> descriptorSets_opt;

    std::vector<VkBuffer> uniformBuffers_opt;
    std::vector<VkDeviceMemory> uniformBuffersMemory_opt;
    std::vector<void*> uniformBuffersMapped_opt;

    std::vector<VkDescriptorSet> computeDescriptorSets;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    VkImage raymarchingDepthImage;
    VkDeviceMemory raymarchingDepthImageMemory;
    VkImageView raymarchingDepthImageView;

    glm::vec3 position;
    glm::vec3 looking_at;
    float x_angle = 0.0;
    float y_angle = 0.0;
    glm::mat4 x_rotation;
    glm::mat4 y_rotation;
    float sensivity = 0.001;
    float speed = 2.0;
    double cursorX = 0.0;
    double cursorY = 0.0;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    std::vector<VkFence> computeInFlightFences;
    std::vector<VkSemaphore> computeFinishedSemaphores;
    std::vector<VkSemaphore> sortStageFinishedSemaphores;
    std::vector<VkSemaphore> statendFinishedSemaphores;
    std::vector<VkSemaphore> preRaymarchingFinishedSemaphores;

    std::chrono::high_resolution_clock::time_point timeOfLastUpdate;

    std::vector<float> computeTimers;
    std::vector<float> renderTimers;
    std::vector<float> sortTimers;
    std::vector<float> statendTimers;
    std::vector<float> assignTimers;

    uint32_t currentTimePoint = 0;

    float computeTime = 0;
    float renderTime = 0;

    float timePeriod;

    VkQueryPool timeQueryPool;

    bool cursorMode;
    bool isSpacePressed;

    std::vector<VkBuffer> shaderStorageBuffers;
    std::vector<VkDeviceMemory> shaderStorageBuffersMemory;

    VkBuffer shaderStorageBuffersTriangles;
    std::vector<VkBuffer>  shaderStorageBuffersSommets;

    VkDeviceMemory shaderStorageBuffersTrianglesMemory;
    std::vector<VkDeviceMemory> shaderStorageBuffersSommetsMemory;

    std::vector<VkBuffer> chunkBuffers;
    std::vector<VkDeviceMemory> chunkBuffersMemory;
    std::vector<void*> chunkBuffersMapped;

    std::vector<VkBuffer> particlesIndexBuffers;
    std::vector<VkDeviceMemory> particlesIndexBuffersMemory;
    std::vector<void*> particlesIndexBuffersMapped;

    std::vector<VkBuffer> chunkIndexBuffers;
    std::vector<VkDeviceMemory> chunkIndexBuffersMemory;
    std::vector<void*> chunkIndexBuffersMapped;

    std::vector<VkBuffer> sortBuffers;
    std::vector<VkDeviceMemory> sortBuffersMemory;


    std::vector<Swap> swaps;

    VkBuffer swapBuffer;
    VkDeviceMemory swapBufferMemory;

    std::vector<uint32_t> chunks;
    std::vector<uint32_t> particlesIndex;
    std::vector<Statend> chunksIndex;

    size_t currentFrame = 0;
    float deltaTime = 1. ;

    uint turn = 0;

    uint nb_sort_stages = 1;

    std::vector<SortBufferObject> taille_swaps;

    std::vector<std::vector<VkSubmitInfo>> computeSubmitInfo;
    VkPipelineStageFlags waitStagesGeneric[1] = { VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT };

    OptionBufferObject opt = {
        0.30, //l0
        1.0, //k
        0.1, //alpha
        0.98, //g
        0.013, //dt
        1.0, //max_dist
        static_cast<uint>(perles.size()), //nb particules
        static_cast<uint>(indices.size()/3), //nb triangles
        static_cast<uint>(vertices.size()),  //nb points
        50*STATEND_STEP, //chunk_nb
        2.0, //color
        0.001,  // epsilon
        0.015,//blending
    };

    void initWindow(){
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        const char* description;
        int code = glfwGetError(&description);
 
        if (description)
            printf("\nerror : %s \n\n",description);
        assert(window != NULL);
    }

    void createInstance() {

        //on tente de voir si on peut débuger
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }   

        //ça c'est l'init de Vk
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // des paramètres venus de glfw pour les extensions dont il a besoin
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        const char** totalExtensions;
        totalExtensions = new const char*[glfwExtensionCount+2];
        for(int i =0; i < glfwExtensionCount; i++)
            totalExtensions[i] = glfwExtensions[i];
        totalExtensions[glfwExtensionCount] = "VK_KHR_get_physical_device_properties2";
        totalExtensions[glfwExtensionCount+1] =  "VK_EXT_debug_utils";

        createInfo.enabledExtensionCount = glfwExtensionCount+1;
        createInfo.ppEnabledExtensionNames = totalExtensions;
        

        //Check pour les extensions nésséssaires et optionnelles
        uint32_t extensionCount = 0;

        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        std::cout << "available extensions:\n";

        for (const auto& extension : extensions) {
            std::cout << '\t' << extension.extensionName << '\n';
        }

        //GROS debug. debug sur le debug.
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }


        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }

        //check pour les extensions de glfw : 
        
        for (int i_glfw = 0; i_glfw < glfwExtensionCount; i_glfw++) {
            bool cond = false;
            for (const auto& extension : extensions) {
                if(strcmp(extension.extensionName,glfwExtensions[i_glfw]) == 0)
                    cond = true;
            }
            if(!cond)
            {
                std::cout << "failed to get extension " << glfwExtensions[i_glfw] << " from glfw. \n";
                throw std::runtime_error("Failed to get a glfw extension");
            }

        }
        std::cout << "glfw extensions successfully initialized \n";
        delete[] totalExtensions;

    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;
        // Code pour trouver les indices de familles à ajouter à la structure

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        
        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) 
                && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                indices.graphicsAndComputeFamily = i;
            }
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if(presentSupport)
            {
                indices.presentFamily = i;
            }
            if (indices.isComplete()) {
                break;
            }
            i++;
        }
        return indices;
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {

        //pas utile pour le moment mais bien de savoir que ça existe 
        
        //ce qu'est notre carte graphique
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);

        //ce qu'elle peut faire
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        VkPhysicalDeviceHostQueryResetFeatures resetFeatures;
        resetFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES;

        VkPhysicalDeviceFeatures2 deviceFeatures2;
        deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        deviceFeatures2.pNext = &resetFeatures;
        vkGetPhysicalDeviceFeatures2(device, &deviceFeatures2);
        
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }


        return indices.isComplete() && extensionsSupported && swapChainAdequate 
        && (resetFeatures.hostQueryReset == VK_TRUE) && (deviceProperties.limits.timestampComputeAndGraphics == VK_TRUE)
            && (deviceProperties.limits.timestampPeriod > 0);
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());



        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("aucun GPU ne peut exécuter ce programme!");
        }
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    void createLogicalDevice() {

        // Une interface avec la carte graphique !

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = 
            {indices.graphicsAndComputeFamily.value(), indices.presentFamily.value()};

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        //pour mesurer du temps !
        VkPhysicalDeviceHostQueryResetFeatures resetFeatures;
        resetFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES;
        resetFeatures.pNext = nullptr;
        resetFeatures.hostQueryReset = VK_TRUE;

        createInfo.pNext = &resetFeatures;

        
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("échec lors de la création d'un logical device!");
        }
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
        vkGetDeviceQueue(device, indices.graphicsAndComputeFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.graphicsAndComputeFamily.value(), 0, &computeQueue);
    }   

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("échec de la création de la window surface!");
        }
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {

        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && 
                availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes) {
        /*for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }*/


        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {

        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            VkExtent2D actualExtent = {WIDTH, HEIGHT};

            actualExtent.width = std::clamp(actualExtent.width, 
                    capabilities.minImageExtent.width, 
                    capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, 
                    capabilities.minImageExtent.height, 
                    capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

    
        return details;
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        if (swapChainSupport.capabilities.maxImageCount > 0 
            && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }
        printf("max images in swap chain = %d\n", imageCount);
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsAndComputeFamily.value(), indices.presentFamily.value()};

        if (indices.graphicsAndComputeFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0; // Optionnel
            createInfo.pQueueFamilyIndices = nullptr; // Optionnel
        }
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("échec de la création de la swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews() {

        swapChainImageViews.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("échec de la création d'une image view!");
            }
        }
    }

    static std::vector<char> readFile(const std::string& filename) {

        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error(std::string {"échec de l'ouverture du fichier "} + filename + "!");
        }
        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {

        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("échec de la création d'un module shader!");
        }

        return shaderModule;
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;


        std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcAccessMask = 0;

        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT 
        | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT 
        | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT 
        | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("échec de la création de la render pass!");
        }

    }

    void createGraphicsPipeline() {
        
        auto vertShaderCode = readFile("shaders/vert.spv");
        auto fragShaderCode = readFile("shaders/frag.spv");

        auto vertShaderModule = createShaderModule(vertShaderCode);
        auto fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";
        vertShaderStageInfo.pSpecializationInfo = nullptr;

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapChainExtent.width;
        viewport.height = (float) swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT; //VK_CULL_MODE_NONE
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_TRUE;
        rasterizer.depthBiasConstantFactor = 0.0f; // Optionnel
        rasterizer.depthBiasClamp = 0.0f; // Optionnel
        rasterizer.depthBiasSlopeFactor = 0.0f; // Optionnel

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f; // Optionnel
        multisampling.pSampleMask = nullptr; // Optionnel
        multisampling.alphaToCoverageEnable = VK_FALSE; // Optionnel
        multisampling.alphaToOneEnable = VK_FALSE; // Optionnel

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT 
            | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optionnel
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optionnel
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optionnel
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optionnel
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optionnel
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optionnel

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f; // Optional
        depthStencil.maxDepthBounds = 1.0f; // Optional
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {}; // Optional
        depthStencil.back = {}; // Optional

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optionnel
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // Optionnel
        colorBlending.blendConstants[1] = 0.0f; // Optionnel
        colorBlending.blendConstants[2] = 0.0f; // Optionnel
        colorBlending.blendConstants[3] = 0.0f; // Optionnel

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_LINE_WIDTH
        };

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;       // Optionnel
        pipelineLayoutInfo.pushConstantRangeCount = 0;    // Optionnel
        pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optionnel

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("échec de la création du pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = nullptr; // Optionnel
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optionnel
        pipelineInfo.basePipelineIndex = -1; // Optionnel

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) 
                != VK_SUCCESS) {
            throw std::runtime_error("échec de la création de la pipeline graphique!");
        }

        printf("pipeline succefully created !\n");

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createRaymarchingPipeline() {
        
        auto vertShaderCode = readFile("shaders/vert.spv");
        auto fragShaderCode = readFile("shaders/march.spv");

        auto vertShaderModule = createShaderModule(vertShaderCode);
        auto fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";
        vertShaderStageInfo.pSpecializationInfo = nullptr;

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapChainExtent.width;
        viewport.height = (float) swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT; //VK_CULL_MODE_NONE
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_TRUE;
        rasterizer.depthBiasConstantFactor = 0.0f; // Optionnel
        rasterizer.depthBiasClamp = 0.0f; // Optionnel
        rasterizer.depthBiasSlopeFactor = 0.0f; // Optionnel

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f; // Optionnel
        multisampling.pSampleMask = nullptr; // Optionnel
        multisampling.alphaToCoverageEnable = VK_FALSE; // Optionnel
        multisampling.alphaToOneEnable = VK_FALSE; // Optionnel

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT 
        | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optionnel
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optionnel
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optionnel
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optionnel
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optionnel
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optionnel

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f; // Optional
        depthStencil.maxDepthBounds = 1.0f; // Optional
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {}; // Optional
        depthStencil.back = {}; // Optional

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optionnel
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // Optionnel
        colorBlending.blendConstants[1] = 0.0f; // Optionnel
        colorBlending.blendConstants[2] = 0.0f; // Optionnel
        colorBlending.blendConstants[3] = 0.0f; // Optionnel

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_LINE_WIDTH
        };

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;       // Optionnel
        pipelineLayoutInfo.pushConstantRangeCount = 0;    // Optionnel
        pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optionnel

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &raymarchingPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("échec de la création du pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = nullptr; // Optionnel
        pipelineInfo.layout = raymarchingPipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optionnel
        pipelineInfo.basePipelineIndex = -1; // Optionnel

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &raymarchingPipeline)
             != VK_SUCCESS) {
            throw std::runtime_error("échec de la création de la pipeline graphique!");
        }

        printf("pipeline succefully created !\n");

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createComputePipeline(VkPipelineLayout& argPipelineLayout, VkPipeline& argPipeline,const char* shaderFile ){
        auto computeShaderCode = readFile(shaderFile);

        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &argPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = argPipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &argPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }

    void createSortPipeline(){
        auto computeShaderCode = readFile("shaders/sort.spv");

        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &sortingDescriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &sortPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = sortPipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &sortPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {

            std::array<VkImageView, 2> attachments = {
                swapChainImageViews[i],
                depthImageView
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("échec de la création d'un framebuffer!");
            }
        }
    }

    void createRaymarchingFramebuffers() {
        raymarchingFramebuffers.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {

            std::array<VkImageView, 2> attachments = {
                raymarchingImageViews[i],
                raymarchingDepthImageView
            };



            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &raymarchingFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("échec de la création d'un framebuffer!");
            }
        }
    }

    void createRaymarchingImages(){

        raymarchingImageViews.resize(MAX_FRAMES_IN_FLIGHT);
        raymarchingImages.resize(MAX_FRAMES_IN_FLIGHT);
        raymarchingImagesMemory.resize(MAX_FRAMES_IN_FLIGHT);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {

            createImage(swapChainExtent.width, swapChainExtent.height, 
            VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_TILING_OPTIMAL, 
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
            raymarchingImages[i], raymarchingImagesMemory[i]
            );

            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = raymarchingImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &createInfo, nullptr, &raymarchingImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("échec de la création d'une image view pour raymarching!");
            }
        }
    }

    void createRaymarchingDepthResources() {

        VkFormat depthFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
        createImage(swapChainExtent.width, swapChainExtent.height, 
            depthFormat, 
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
            raymarchingDepthImage, raymarchingDepthImageMemory
        );
        raymarchingDepthImageView = createImageView(raymarchingDepthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    void createDepthResources() {

        VkFormat depthFormat = findDepthFormat();
        createImage(swapChainExtent.width, swapChainExtent.height, 
            depthFormat, 
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
            depthImage, depthImageMemory
        );
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    void createTextureSampler(){

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;

        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 1;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;

        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; 

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("échec de la création d'une command pool!");
        }
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) 
            {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, 
            VkBuffer& buffer, VkDeviceMemory& bufferMemory) {

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    void createParticleBuffer(){

        shaderStorageBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        shaderStorageBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        VkDeviceSize bufferSize = sizeof(perles[0]) * perles.size();
        VkBufferCreateInfo bufferInfo{};

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, perles.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, 
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT 
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
                shaderStorageBuffers[i], 
                shaderStorageBuffersMemory[i]);
            // Copy data from the staging buffer (host) to the shader storage buffer (GPU)
            copyBuffer(stagingBuffer, shaderStorageBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createSommetBuffer(){

        shaderStorageBuffersSommets.resize(MAX_FRAMES_IN_FLIGHT);
        shaderStorageBuffersSommetsMemory.resize(MAX_FRAMES_IN_FLIGHT);

        VkDeviceSize bufferSize = sizeof(end_vertices[0]) * end_vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, end_vertices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, 
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
                    shaderStorageBuffersSommets[i], 
                    shaderStorageBuffersSommetsMemory[i]);
            // Copy data from the staging buffer (host) to the shader storage buffer (GPU)
            copyBuffer(stagingBuffer, shaderStorageBuffersSommets[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createSwapBuffer(){

        VkDeviceSize bufferSize = sizeof(swaps[0]) * swaps.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, swaps.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, 
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
                swapBuffer, 
                swapBufferMemory);
        // Copy data from the staging buffer (host) to the shader storage buffer (GPU)
        copyBuffer(stagingBuffer, swapBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createTriangleBuffer(){

        VkDeviceSize bufferSize = sizeof(uint32_t) * end_indices.size();
        VkBufferCreateInfo bufferInfo{};

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, end_indices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, 
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
                shaderStorageBuffersTriangles, 
                shaderStorageBuffersTrianglesMemory);
        // Copy data from the staging buffer (host) to the shader storage buffer (GPU)
        copyBuffer(stagingBuffer, shaderStorageBuffersTriangles, bufferSize);


        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createVertexBuffer(){
        VkDeviceSize bufferSize = sizeof(end_vertices[0]) * end_vertices.size();
        VkBufferCreateInfo bufferInfo{};

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, end_vertices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
    //modifié pour la science !!!
    void createIndexBuffer() {
        VkDeviceSize bufferSize;
        if(JUST_RAY_MARCHING) 
            bufferSize = sizeof(uint32_t) * indices.size();
        else bufferSize = sizeof(uint32_t) * end_indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        if(JUST_RAY_MARCHING) 
            memcpy(data, indices.data(), (size_t) bufferSize);
         else memcpy(data, end_indices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, 
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createChunkBuffer(){


        chunkBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        chunkBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        //chunkBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        VkDeviceSize bufferSize = sizeof(int32_t) * opt.nb_particles;
        VkBufferCreateInfo bufferInfo{};

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, 
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT ,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, /* VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT 
                | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,*/
                chunkBuffers[i], 
                chunkBuffersMemory[i]);
            /*if(vkMapMemory(device, chunkBuffersMemory[i], 0, bufferSize, 0, &chunkBuffersMapped[i]) != VK_SUCCESS)
                throw std::runtime_error("la mémoire elle veux pas lol");*/
        }
    }

    void createSortBuffers(){
        sortBuffers.resize(nb_sort_stages);
        sortBuffersMemory.resize(nb_sort_stages);

        VkDeviceSize bufferSize = sizeof(SortBufferObject);

        for(int i = 0; i < nb_sort_stages; i++){
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;

            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            stagingBuffer, stagingBufferMemory);

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, &taille_swaps[i], (size_t) bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            createBuffer(bufferSize, 
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
                    sortBuffers[i], 
                    sortBuffersMemory[i]);
            // Copy data from the staging buffer (host) to the shader storage buffer (GPU)
            copyBuffer(stagingBuffer, sortBuffers[i], bufferSize);

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }
    }

    void createParticlesIndexBuffer(){

        particlesIndexBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        particlesIndexBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        //particlesIndexBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        VkDeviceSize bufferSize = sizeof(uint32_t) * opt.nb_particles;

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, particlesIndex.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, 
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, /* VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT 
                | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,*/
                particlesIndexBuffers[i], 
                particlesIndexBuffersMemory[i]);
            // Copy data from the staging buffer (host) to the shader storage buffer (GPU)
            copyBuffer(stagingBuffer, particlesIndexBuffers[i], bufferSize);
            /*if(vkMapMemory(device, particlesIndexBuffersMemory[i], 0, bufferSize, 0, &particlesIndexBuffersMapped[i]) != VK_SUCCESS)
                throw std::runtime_error("la mémoire elle veux pas lol");*/
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createChunkIndexBuffer(){

        chunkIndexBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        chunkIndexBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        //chunkIndexBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        VkDeviceSize bufferSize = sizeof(Statend) * opt.nb_chunks;
        //std::cout << "buffer size : " << bufferSize << "\n";

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, 
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, /* VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT 
                | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,*/
                chunkIndexBuffers[i], 
                chunkIndexBuffersMemory[i]);
            /*if(vkMapMemory(device, chunkIndexBuffersMemory[i], 0, bufferSize, 0, &chunkIndexBuffersMapped[i]) 
                != VK_SUCCESS)
                throw std::runtime_error("la mémoire elle veux pas lol");*/
        }
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0; // Optional
        copyRegion.dstOffset = 0; // Optional
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
        vkEndCommandBuffer(commandBuffer);
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void recordComputeCommandBuffers(){
        for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            if (vkBeginCommandBuffer(computeCommandBuffers[i], &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }

            vkCmdBindPipeline(computeCommandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
            vkCmdBindDescriptorSets(computeCommandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, 
                computePipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

            vkCmdResetQueryPool(computeCommandBuffers[i], timeQueryPool, i*8, 2);

            vkCmdWriteTimestamp(computeCommandBuffers[i], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timeQueryPool, i*8); 

            vkCmdDispatch(computeCommandBuffers[i], opt.nb_particles /COMPUTE_STEP , 1, 1);

            vkCmdWriteTimestamp(computeCommandBuffers[i], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timeQueryPool, i*8 +1); 

            if (vkEndCommandBuffer(computeCommandBuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0; // Optionnel
        beginInfo.pInheritanceInfo = nullptr; // Optionel

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("erreur au début de l'enregistrement d'un command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = raymarchingFramebuffers[currentFrame];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;

        vkCmdResetQueryPool(commandBuffer, timeQueryPool, currentFrame*8 + 2, 2);
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timeQueryPool, currentFrame*8+2); 

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, 
            pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        VkBuffer vertexBuffers[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};

        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &shaderStorageBuffersSommets[currentFrame], offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);


        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(end_indices.size()) - 
                static_cast<uint32_t>(indices.size()), 1, static_cast<uint32_t>(indices.size()), 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timeQueryPool, currentFrame*8 +3); 

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("échec de l'enregistrement d'un command buffer!");
        }
    }

    void recordRaymarchingCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0; // Optionnel
        beginInfo.pInheritanceInfo = nullptr; // Optionel

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("erreur au début de l'enregistrement d'un command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;

        //vkCmdResetQueryPool(commandBuffer, timeQueryPool, currentFrame*8 + 2, 2);
        //vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timeQueryPool, currentFrame*8+2); 

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, raymarchingPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, 
            raymarchingPipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        VkBuffer vertexBuffers[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};

        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &shaderStorageBuffersSommets[currentFrame], offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        if(JUST_RAY_MARCHING)
            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        else
            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(end_indices.size()), 1, 0, 0, 0);
        vkCmdEndRenderPass(commandBuffer);

        //vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timeQueryPool, currentFrame*8 +3); 

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("échec de l'enregistrement d'un command buffer!");
        }
    }

    void recordSortCommandBuffers(){
        for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            for(int j = 0; j < nb_sort_stages; j++)
            {
                VkCommandBufferBeginInfo beginInfo{};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

                if (vkBeginCommandBuffer(sortCommandBuffers[i*nb_sort_stages + j], &beginInfo) != VK_SUCCESS) {
                    throw std::runtime_error("failed to begin recording command buffer!");
                }

                vkCmdBindPipeline(sortCommandBuffers[i*nb_sort_stages + j], VK_PIPELINE_BIND_POINT_COMPUTE, 
                    sortPipeline);
                vkCmdBindDescriptorSets(sortCommandBuffers[i*nb_sort_stages + j], VK_PIPELINE_BIND_POINT_COMPUTE, 
                    sortPipelineLayout, 0, 1, &sortingDescriptorSets[i*nb_sort_stages + j], 0, nullptr);

                vkCmdDispatch(sortCommandBuffers[i*nb_sort_stages + j], 
                        std::ceil(taille_swaps[j].nb_swaps/SORT_STEP) , 1, 1);

                if (vkEndCommandBuffer(sortCommandBuffers[i*nb_sort_stages + j]) != VK_SUCCESS) {
                    throw std::runtime_error("failed to record command buffer!");
                }
            }
        }
    }

    void recordAssignCommandBuffers(){
        for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            if (vkBeginCommandBuffer(assignCommandBuffers[i], &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }

            vkCmdBindPipeline(assignCommandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, assignPipeline);
            vkCmdBindDescriptorSets(assignCommandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, 
                assignPipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

            vkCmdResetQueryPool(assignCommandBuffers[i], timeQueryPool, i*8+4, 2);

            vkCmdWriteTimestamp(assignCommandBuffers[i], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timeQueryPool, i*8+4);  

            vkCmdDispatch(assignCommandBuffers[i], std::ceil(opt.nb_particles /ASSIGN_STEP) , 1, 1);

            vkCmdWriteTimestamp(assignCommandBuffers[i], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timeQueryPool, i*8+5);

            if (vkEndCommandBuffer(assignCommandBuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }
    }

    void recordStatendCommandBuffers(){
        for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            if (vkBeginCommandBuffer(statendCommandBuffers[i], &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }

            vkCmdBindPipeline(statendCommandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, statendPipeline);
            vkCmdBindDescriptorSets(statendCommandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, 
                statendPipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

            vkCmdResetQueryPool(statendCommandBuffers[i], timeQueryPool, i*8+6, 2);

            vkCmdWriteTimestamp(statendCommandBuffers[i], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timeQueryPool, i*8+6);

            vkCmdDispatch(statendCommandBuffers[i], opt.nb_chunks /STATEND_STEP , 1, 1);

            vkCmdWriteTimestamp(statendCommandBuffers[i], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timeQueryPool, i*8+7);

            if (vkEndCommandBuffer(statendCommandBuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }
    }

    void createCommandBuffers(std::vector<VkCommandBuffer>& cmdBuffer, uint32_t nb_buffers) {

        cmdBuffer.resize(nb_buffers);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = nb_buffers;

        if (vkAllocateCommandBuffers(device, &allocInfo, cmdBuffer.data()) != VK_SUCCESS) {
            throw std::runtime_error("échec de l'allocation de command buffers!");
        }
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
        computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        sortStageFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT*(nb_sort_stages+1));
        statendFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        preRaymarchingFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        VkFenceCreateInfo fenceInfo_off{};
        fenceInfo_off.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS
                ||vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS
                ||vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS
                ||vkCreateSemaphore(device, &semaphoreInfo, nullptr, &computeFinishedSemaphores[i]) != VK_SUCCESS
                ||vkCreateSemaphore(device, &semaphoreInfo, nullptr, &preRaymarchingFinishedSemaphores[i]) != VK_SUCCESS 
                ||vkCreateSemaphore(device, &semaphoreInfo, nullptr, &statendFinishedSemaphores[i]) != VK_SUCCESS ) {

                throw std::runtime_error("échec de la création des objets de synchronisation pour une frame!");
            }
        }
        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT-1; i++) {
            if (vkCreateFence(device, &fenceInfo_off, nullptr, &computeInFlightFences[i]) != VK_SUCCESS)
                throw std::runtime_error("problèmes de fence !!");
        }
        if (vkCreateFence(device, &fenceInfo, nullptr, &computeInFlightFences[MAX_FRAMES_IN_FLIGHT-1]) != VK_SUCCESS)
                throw std::runtime_error("problèmes de fence !!");

        printf("fence signaled : %d. Fence not signaled : %d\n", VK_SUCCESS, VK_NOT_READY);
        for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++){
            printf("fence %d is currently in a %d state \n", i, vkGetFenceStatus(device,computeInFlightFences[i]));

        }


        for(int i = 0; i < MAX_FRAMES_IN_FLIGHT*(nb_sort_stages+1); i++)
            if(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &sortStageFinishedSemaphores[i]) != VK_SUCCESS)
                throw std::runtime_error("échec de la création des objets de synchronisation pour le tri !");

        for(size_t i = 0; i < swapChainImages.size(); i++)
            if(vkCreateFence(device, &fenceInfo, nullptr, &imagesInFlight[i]) != VK_SUCCESS)
                throw std::runtime_error("échec de la création des objets de synchronisation pour une frame!");
    }

    void createDescriptorSetLayout() {

        std::array<VkDescriptorSetLayoutBinding, 10> layoutBindings{};


        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT 
            | VK_SHADER_STAGE_COMPUTE_BIT ;
        layoutBindings[0].pImmutableSamplers = nullptr; // Optional


        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT 
            | VK_SHADER_STAGE_COMPUTE_BIT ;
        layoutBindings[1].pImmutableSamplers = nullptr; // Optional

        //compute thingys
        layoutBindings[2].binding = 2;
        layoutBindings[2].descriptorCount = 1;
        layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[2].pImmutableSamplers = nullptr;
        layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        
        layoutBindings[3].binding = 3;
        layoutBindings[3].descriptorCount = 1;
        layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[3].pImmutableSamplers = nullptr;
        layoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[4].binding = 4;
        layoutBindings[4].descriptorCount = 1;
        layoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[4].pImmutableSamplers = nullptr;
        layoutBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[5].binding = 5;
        layoutBindings[5].descriptorCount = 1;
        layoutBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[5].pImmutableSamplers = nullptr;
        layoutBindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[6].binding = 6;
        layoutBindings[6].descriptorCount = 1;
        layoutBindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[6].pImmutableSamplers = nullptr;
        layoutBindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        layoutBindings[7].binding = 7;
        layoutBindings[7].descriptorCount = 1;
        layoutBindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[7].pImmutableSamplers = nullptr;
        layoutBindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        layoutBindings[8].binding = 8;
        layoutBindings[8].descriptorCount = 1;
        layoutBindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[8].pImmutableSamplers = nullptr;
        layoutBindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        layoutBindings[9].binding = 14;
        layoutBindings[9].descriptorCount = 1;
        layoutBindings[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBindings[9].pImmutableSamplers = nullptr;
        layoutBindings[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 10;
        layoutInfo.pBindings = layoutBindings.data();
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }
    
    void createSortingDescriptorSetLayout(){

        std::array<VkDescriptorSetLayoutBinding, 4> layoutBindings{};

        layoutBindings[0].binding = 10;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT ;
        layoutBindings[0].pImmutableSamplers = nullptr; // Optional

        layoutBindings[1].binding = 11;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        layoutBindings[2].binding = 12;
        layoutBindings[2].descriptorCount = 1;
        layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[2].pImmutableSamplers = nullptr;
        layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        layoutBindings[3].binding = 13;
        layoutBindings[3].descriptorCount = 1;
        layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[3].pImmutableSamplers = nullptr;
        layoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 4;
        layoutInfo.pBindings = layoutBindings.data();
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &sortingDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                uniformBuffers[i], uniformBuffersMemory[i]);

            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }

        //pour les opt : 
        VkDeviceSize bufferSize_opt = sizeof(OptionBufferObject);

        uniformBuffers_opt.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory_opt.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped_opt.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize_opt, 
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                uniformBuffers_opt[i], uniformBuffersMemory_opt[i]);

            vkMapMemory(device, uniformBuffersMemory_opt[i], 0, bufferSize_opt, 0, &uniformBuffersMapped_opt[i]);
        }
    }

    void updateUniformBuffer(uint32_t currentImage) {

        UniformBufferObject ubo{};
        ubo.model = glm::mat4(1.0f);

        ubo.view = glm::lookAt(position, 
            position + looking_at , 
            glm::cross(looking_at, glm::cross( glm::vec3(0.0f, 1.0f, 0.0f), looking_at))
            );
        
        ubo.proj = glm::perspective(glm::radians(90.0f), 
            swapChainExtent.width / (float) swapChainExtent.height, 
            0.1f, 50.0f);
        ubo.proj[1][1] *= -1;
        ubo.cam = position;
        ubo.lightsource = glm::normalize (glm::vec3(0.5,2.0,1.0));
        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));


        opt.delta_t = deltaTime;
        memcpy(uniformBuffersMapped_opt[currentImage], &opt, sizeof(opt));
    }

    void createDescriptorPool() {

        std::array<VkDescriptorPoolSize, 3> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)*2;

        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)*7;
        
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[2].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 3;
        poolInfo.pPoolSizes = poolSizes.data();

        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createSortingDescriptorPool() {

        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)*nb_sort_stages;

        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)*3*nb_sort_stages;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes.data();

        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)*nb_sort_stages;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &sortingDescriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) { 

            std::array<VkWriteDescriptorSet, 10> descriptorWrites{};


            VkDescriptorBufferInfo uniformBufferInfo{};
            uniformBufferInfo.buffer = uniformBuffers[i];
            uniformBufferInfo.offset = 0;
            uniformBufferInfo.range = sizeof(UniformBufferObject);

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;

            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;

            descriptorWrites[0].pBufferInfo = &uniformBufferInfo;
            descriptorWrites[0].pImageInfo = nullptr; // Optional
            descriptorWrites[0].pTexelBufferView = nullptr; // Optional

            VkDescriptorBufferInfo optionBufferInfo{};
            optionBufferInfo.buffer = uniformBuffers_opt[i];
            optionBufferInfo.offset = 0;
            optionBufferInfo.range = sizeof(OptionBufferObject);

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;

            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[1].descriptorCount = 1;

            descriptorWrites[1].pBufferInfo = &optionBufferInfo;
            descriptorWrites[1].pImageInfo = nullptr; // Optional
            descriptorWrites[1].pTexelBufferView = nullptr; // Optional

            VkDescriptorBufferInfo storageBufferInfoLastFrame{};
            if(i == 0)
            {
                storageBufferInfoLastFrame.buffer = shaderStorageBuffers[MAX_FRAMES_IN_FLIGHT-1];
                //printf( "%d %d \n",i,MAX_FRAMES_IN_FLIGHT-1);
            }
            else {
                storageBufferInfoLastFrame.buffer = shaderStorageBuffers[i-1];
                //printf( "%d %d \n",i,i-1);
            }
            storageBufferInfoLastFrame.offset = 0;
            storageBufferInfoLastFrame.range = sizeof(perles[0]) * perles.size();


            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = descriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &storageBufferInfoLastFrame;

            VkDescriptorBufferInfo storageBufferInfoCurrentFrame{};
            storageBufferInfoCurrentFrame.buffer = shaderStorageBuffers[i];
            storageBufferInfoCurrentFrame.offset = 0;
            storageBufferInfoCurrentFrame.range = sizeof(perles[0]) * perles.size();

            descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[3].dstSet = descriptorSets[i];
            descriptorWrites[3].dstBinding = 3;
            descriptorWrites[3].dstArrayElement = 0;
            descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[3].descriptorCount = 1;
            descriptorWrites[3].pBufferInfo = &storageBufferInfoCurrentFrame;

            VkDescriptorBufferInfo storageBufferInfoSommets{};
            storageBufferInfoSommets.buffer = shaderStorageBuffersSommets[i];
            storageBufferInfoSommets.offset = 0;
            storageBufferInfoSommets.range = sizeof(end_vertices[0]) * end_vertices.size();

            descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[4].dstSet = descriptorSets[i];
            descriptorWrites[4].dstBinding = 4;
            descriptorWrites[4].dstArrayElement = 0;
            descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[4].descriptorCount = 1;
            descriptorWrites[4].pBufferInfo = &storageBufferInfoSommets;

            VkDescriptorBufferInfo storageBufferInfoTriangles{};
            storageBufferInfoTriangles.buffer = indexBuffer;
            storageBufferInfoTriangles.offset = 0;
            if(JUST_RAY_MARCHING)
                storageBufferInfoTriangles.range = sizeof(uint32_t) * indices.size();
            else
                storageBufferInfoTriangles.range = sizeof(uint32_t) * end_indices.size();

            descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[5].dstSet = descriptorSets[i];
            descriptorWrites[5].dstBinding = 5;
            descriptorWrites[5].dstArrayElement = 0;
            descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[5].descriptorCount = 1;
            descriptorWrites[5].pBufferInfo = &storageBufferInfoTriangles;

            VkDescriptorBufferInfo storageBufferInfoChunks{};
            storageBufferInfoChunks.buffer = chunkBuffers[i];
            storageBufferInfoChunks.offset = 0;
            storageBufferInfoChunks.range = sizeof(uint32_t) * opt.nb_particles;

            descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[6].dstSet = descriptorSets[i];
            descriptorWrites[6].dstBinding = 6;
            descriptorWrites[6].dstArrayElement = 0;
            descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[6].descriptorCount = 1;
            descriptorWrites[6].pBufferInfo = &storageBufferInfoChunks;

            VkDescriptorBufferInfo storageBufferInfoParticlesIndex{};
            storageBufferInfoParticlesIndex.buffer = particlesIndexBuffers[i];
            storageBufferInfoParticlesIndex.offset = 0;
            storageBufferInfoParticlesIndex.range = sizeof(uint32_t) * opt.nb_particles;

            descriptorWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[7].dstSet = descriptorSets[i];
            descriptorWrites[7].dstBinding = 7;
            descriptorWrites[7].dstArrayElement = 0;
            descriptorWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[7].descriptorCount = 1;
            descriptorWrites[7].pBufferInfo = &storageBufferInfoParticlesIndex;

            VkDescriptorBufferInfo storageBufferInfoChunksIndex{};
            storageBufferInfoChunksIndex.buffer = chunkIndexBuffers[i];
            storageBufferInfoChunksIndex.offset = 0;
            storageBufferInfoChunksIndex.range = sizeof(Statend) * opt.nb_chunks;

            descriptorWrites[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[8].dstSet = descriptorSets[i];
            descriptorWrites[8].dstBinding = 8;
            descriptorWrites[8].dstArrayElement = 0;
            descriptorWrites[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[8].descriptorCount = 1;
            descriptorWrites[8].pBufferInfo = &storageBufferInfoChunksIndex;

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = raymarchingImageViews[currentFrame];
            imageInfo.sampler = textureSampler;

            descriptorWrites[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[9].dstSet = descriptorSets[i];
            descriptorWrites[9].dstBinding = 14;
            descriptorWrites[9].dstArrayElement = 0;
            descriptorWrites[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[9].descriptorCount = 1;
            descriptorWrites[9].pImageInfo = &imageInfo;

            vkUpdateDescriptorSets(device, 10, descriptorWrites.data(), 0, nullptr);
        }
    }

    void createSortingDescriptorSets(){
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT*nb_sort_stages, sortingDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = sortingDescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT*nb_sort_stages);
        allocInfo.pSetLayouts = layouts.data();

        sortingDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT*nb_sort_stages);
        if (vkAllocateDescriptorSets(device, &allocInfo, sortingDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
        for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            for(int j = 0; j < nb_sort_stages; j++)
            {
                std::array<VkWriteDescriptorSet, 4> descriptorWrites{};

                VkDescriptorBufferInfo optionBufferInfo{};
                optionBufferInfo.buffer = sortBuffers[j];
                optionBufferInfo.offset = 0;
                optionBufferInfo.range = sizeof(SortBufferObject);

                descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[0].dstSet = sortingDescriptorSets[i*nb_sort_stages + j];
                descriptorWrites[0].dstBinding = 10;
                descriptorWrites[0].dstArrayElement = 0;
                descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                descriptorWrites[0].descriptorCount = 1;
                descriptorWrites[0].pBufferInfo = &optionBufferInfo;
                descriptorWrites[0].pImageInfo = nullptr; // Optional
                descriptorWrites[0].pTexelBufferView = nullptr; // Optional

                VkDescriptorBufferInfo storageBufferInfoChunks{};
                storageBufferInfoChunks.buffer = chunkBuffers[i];
                storageBufferInfoChunks.offset = 0;
                storageBufferInfoChunks.range = sizeof(uint32_t) * opt.nb_particles;

                descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[1].dstSet = sortingDescriptorSets[i*nb_sort_stages + j];
                descriptorWrites[1].dstBinding = 11;
                descriptorWrites[1].dstArrayElement = 0;
                descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                descriptorWrites[1].descriptorCount = 1;
                descriptorWrites[1].pBufferInfo = &storageBufferInfoChunks;

                VkDescriptorBufferInfo storageBufferInfoParticlesIndex{};
                storageBufferInfoParticlesIndex.buffer = particlesIndexBuffers[i];
                storageBufferInfoParticlesIndex.offset = 0;
                storageBufferInfoParticlesIndex.range = sizeof(uint32_t) * opt.nb_particles;

                descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[2].dstSet = sortingDescriptorSets[i*nb_sort_stages + j];
                descriptorWrites[2].dstBinding = 12;
                descriptorWrites[2].dstArrayElement = 0;
                descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                descriptorWrites[2].descriptorCount = 1;
                descriptorWrites[2].pBufferInfo = &storageBufferInfoParticlesIndex;

                VkDescriptorBufferInfo storageBufferInfoSwaps{};
                storageBufferInfoSwaps.buffer = swapBuffer;
                storageBufferInfoSwaps.offset = 0;
                storageBufferInfoSwaps.range = sizeof(Swap) * swaps.size();

                descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[3].dstSet = sortingDescriptorSets[i*nb_sort_stages + j];
                descriptorWrites[3].dstBinding = 13;
                descriptorWrites[3].dstArrayElement = 0;
                descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                descriptorWrites[3].descriptorCount = 1;
                descriptorWrites[3].pBufferInfo = &storageBufferInfoSwaps;

                vkUpdateDescriptorSets(device, 4, descriptorWrites.data(), 0, nullptr);

            }
        }
    }

    void initCamera() {

        position = glm::vec3( 20.0f, 0.0f, 0.0f);
        x_angle = -0.5;
        y_angle = 0.0;
        looking_at = glm::vec3(
                sin(M_PI*x_angle)*cos(M_PI*y_angle),
                sin(M_PI*y_angle),
                cos(M_PI*x_angle)*cos(M_PI*y_angle)
        );
        timeOfLastUpdate = std::chrono::high_resolution_clock::now();
        if (glfwRawMouseMotionSupported())
            glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
        cursorMode = false;
        isSpacePressed = false;
    }

    void moveCamera() {
        if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            position += glm::vec3( speed*deltaTime*sin(M_PI*x_angle), 0.0f, speed*deltaTime*cos(M_PI*x_angle));
        if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            position += glm::vec3( -speed*deltaTime*cos(M_PI*x_angle), 0.0f, speed*deltaTime*sin(M_PI*x_angle));
        if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            position += glm::vec3( -speed*deltaTime*sin(M_PI*x_angle), 0.0f, -speed*deltaTime*cos(M_PI*x_angle));
        if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            position += glm::vec3( speed*deltaTime*cos(M_PI*x_angle), 0.0f, -speed*deltaTime*sin(M_PI*x_angle));
        if(glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            position += glm::vec3(0.0f, speed*deltaTime, 0.0f);
        if(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            position += glm::vec3(0.0f, -speed*deltaTime, 0.0f);

        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        if(glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS && !isSpacePressed)
        {
            if(cursorMode) 
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            else 
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

            cursorMode = !cursorMode;
            isSpacePressed = true;
            cursorX = xpos;
            cursorY = ypos;

        }
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_RELEASE)
            isSpacePressed = false;
        if(cursorMode)
        {

            x_angle += (cursorX - xpos)*sensivity;
            y_angle += (cursorY - ypos)*sensivity;

            if(y_angle > 0.49) y_angle = 0.49;
            if(y_angle < -0.49) y_angle = -0.49;

            while(x_angle > 1.) x_angle = x_angle - 2.;
            while(x_angle < -1.) x_angle = x_angle + 2.;

            cursorX = xpos;
            cursorY = ypos;

            looking_at = glm::vec3(
                sin(M_PI*x_angle)*cos(M_PI*y_angle),
                sin(M_PI*y_angle),
                cos(M_PI*x_angle)*cos(M_PI*y_angle)
                );

        }
    }

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, 
        VkFormatFeatureFlags features) {

        for (VkFormat format : candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    VkFormat findDepthFormat() {
        return findSupportedFormat(
            {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }

    bool hasStencilComponent(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    void createImage(uint32_t width, uint32_t height, VkFormat format,VkImageLayout layout, VkImageTiling tiling, 
        VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = layout;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        viewInfo.subresourceRange.aspectMask = aspectFlags;

        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture image view!");
        }

        return imageView;
    }

    float randomFloat(){
        float rf =  (float)(rand()) / (float)(RAND_MAX);
        return(2.0*rf - 1.0);
    }

    void setParticuleNumber(int nb){
        opt.nb_particles = nb;
        perles.resize(nb);
        particlesIndex.resize(nb);
        chunks.resize(nb);
        for(int i = 0; i < nb; i++)
        {
            float a = randomFloat();
            float b = randomFloat();
            float c = randomFloat();

            perles[i].position = glm::vec3(a,b,c);
            perles[i].velocity = glm::vec3(
                5.0*randomFloat(),
                5.0*randomFloat(),
                5.0*randomFloat()
                );
            particlesIndex[i] = i;
        }
    }

    void initChunkIndex(){
        chunksIndex.resize(opt.nb_chunks);
    }

    void updateVertices(){
        std::vector<glm::vec3> cone = {
            {0.0,0.0,1.0},
            {0.0,0.0,-1.0},
            {0.0,1.0,0.0},
            {0.0,-1.0,0.0},
            {1.0,0.0,0.0},
            {-1.0,0.0,0.0}
        };
        for(int i = vertices.size(); i < end_vertices.size(); i++)
        {
            int k = i - vertices.size();
            end_vertices[i].pos = perles[k/6].position + (opt.l0/3)*cone[k % 6];
        }
    }

    void setVertices(){

        end_vertices.resize(vertices.size() + 6*opt.nb_particles);
        end_indices.resize(indices.size() + 3*8*opt.nb_particles);

        std::vector<glm::vec3> cone = {
            {0.0,0.0,1.0},
            {0.0,0.0,-1.0},
            {0.0,1.0,0.0},
            {0.0,-1.0,0.0},
            {1.0,0.0,0.0},
            {-1.0,0.0,0.0}
        };

        for(int i = 0; i < vertices.size(); i++)
        {
            end_vertices[i].pos = vertices[i].pos;
            end_vertices[i].color = vertices[i].color;
        }
        
        for(int i = vertices.size(); i < end_vertices.size(); i++)
        {
            int k = i - vertices.size();
            end_vertices[i].pos = perles[k/6].position +  (opt.l0/3)*cone[k % 6];
            end_vertices[i].color = glm::vec3(0.0,1.0,1.0);
        }
        
        for(int i = 0; i < perles.size(); i++)
        {
            int k = 6*i + vertices.size();
            int j = i*24 + indices.size();
            end_indices[j] = k ;
            end_indices[j+1] = k + 2;
            end_indices[j+2] = k + 4;

            end_indices[j+3] = k + 2;
            end_indices[j+4] = k ;
            end_indices[j+5] = k + 5;

            end_indices[j+6] = k + 4;
            end_indices[j+7] = k + 3;
            end_indices[j+8] = k ;

            end_indices[j+9] = k +3;
            end_indices[j+10] = k+5;
            end_indices[j+11] = k ;

            end_indices[j+12] = k +1;
            end_indices[j+13] = k +2;
            end_indices[j+14] = k +5;

            end_indices[j+15] = k +2;
            end_indices[j+16] = k +1;
            end_indices[j+17] = k +4;

            end_indices[j+18] = k + 1;
            end_indices[j+19] = k + 3;
            end_indices[j+20] = k + 4;

            end_indices[j+21] = k + 3;
            end_indices[j+22] = k + 1;
            end_indices[j+23] = k + 5;
            /* code pour débugger les collision je laisse ça là au cas ou
            end_indices[j] = k ;
            end_indices[j+1] = k + 5;
            end_indices[j+2] = k + 4;

            end_indices[j+3] = k + 4;
            end_indices[j+4] = k + 1;
            end_indices[j+5] = k + 5;

            end_indices[j+6] = k + 4 ;
            end_indices[j+7] = k + 2;
            end_indices[j+8] = k + 5;

            end_indices[j+9] = k +3;
            end_indices[j+10] = k +5;
            end_indices[j+11] = k + 4;

            end_indices[j+12] = k +1;
            end_indices[j+13] = k +2;
            end_indices[j+14] = k +3;

            end_indices[j+15] = k ;
            end_indices[j+16] = k ;
            end_indices[j+17] = k ;

            end_indices[j+18] = k ;
            end_indices[j+19] = k ;
            end_indices[j+20] = k ;

            end_indices[j+21] = k ;
            end_indices[j+22] = k ;
            end_indices[j+23] = k ; */

        }
        for (int i = 0; i < indices.size(); i++)
            end_indices[i] = indices[i];
    }

    void initTimers(){
        computeTimers.resize(NB_TIMEPOINTS);
        sortTimers.resize(NB_TIMEPOINTS);
        statendTimers.resize(NB_TIMEPOINTS);
        assignTimers.resize(NB_TIMEPOINTS);
        renderTimers.resize(NB_TIMEPOINTS);

        for(int i =0; i < NB_TIMEPOINTS; i++)
        {
            computeTimers[i] = 0.0;
            renderTimers[i] = 0.0;
            sortTimers[i] = 0.0;
            statendTimers[i] = 0.0;
            assignTimers[i] = 0.0;
        }

        VkPhysicalDeviceProperties props;

        vkGetPhysicalDeviceProperties(physicalDevice, &props);

        timePeriod = props.limits.timestampPeriod;
        printf("timestamp period = %f nanoseconds \n", timePeriod);
    }

    void CreateQueryPool(){
        VkQueryPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        createInfo.pNext = nullptr; // Optional
        createInfo.flags = 0; // Reserved for future use, must be 0!

        createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        createInfo.queryCount = MAX_FRAMES_IN_FLIGHT * 8; // REVIEW

        VkResult result = vkCreateQueryPool(device, &createInfo, nullptr, &timeQueryPool);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create time query pool!");
        }

        VkCommandBuffer commandBuffer;
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        vkCmdResetQueryPool(commandBuffer, timeQueryPool, 0, 8*MAX_FRAMES_IN_FLIGHT);

        for(int i = 0; i < 8*MAX_FRAMES_IN_FLIGHT; i++)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timeQueryPool, i);

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void printTimers(){
        float totalComputeTime = 0.0;
        float totalRenderTime= 0.0;
        float totalAssignTime = 0.0;
        float totalSortTime = 0.0;
        float totalStatendTime = 0.0;

        for(int i = 0; i < NB_TIMEPOINTS; i++)
        {
            totalRenderTime += renderTimers[i];
            totalComputeTime += computeTimers[i];
            totalSortTime += sortTimers[i];
            totalStatendTime += statendTimers[i];
            totalAssignTime += assignTimers[i];
        }
        //printf("dt = %4f \n compute/render = %4f/%4f \n",1000*deltaTime,
        //totalComputeTime/NB_TIMEPOINTS,totalRenderTime/NB_TIMEPOINTS);
        printf("| %2.2f |-| %2.4f || %2.2f || %2.5f || %2.2f |-| %2.2f |\n",
            1000*deltaTime, 
            totalAssignTime/NB_TIMEPOINTS,
            totalSortTime/NB_TIMEPOINTS, 
            totalStatendTime/NB_TIMEPOINTS, 
            totalComputeTime/NB_TIMEPOINTS, 

            totalRenderTime/NB_TIMEPOINTS
            );
    }

    void FetchRenderTimeResults(uint32_t i){
        uint64_t timeBuffer[8];

        VkResult result = vkGetQueryPoolResults(device, timeQueryPool, i * 8, 8, sizeof(uint64_t) * 8,
            timeBuffer, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        //printf("result of time fetch is %d\n",result);
        if (result == VK_SUCCESS)
        {
            computeTimers[currentTimePoint] = timePeriod*(timeBuffer[1] - timeBuffer[0]) / 1000000;
            /*conversion nanoseconde -> milliseconde*/
            renderTimers[currentTimePoint] = timePeriod*(timeBuffer[3] - timeBuffer[2]) / 1000000;

            assignTimers[currentTimePoint] = timePeriod*(timeBuffer[5] - timeBuffer[4]) / 1000000;

            statendTimers[currentTimePoint] = timePeriod*(timeBuffer[7] - timeBuffer[6]) / 1000000;

            sortTimers[currentTimePoint] = timePeriod*(timeBuffer[6] - timeBuffer[5]) / 1000000; 
            //approximation ~ ~ 

            currentTimePoint += 1;
            if(currentTimePoint == NB_TIMEPOINTS)
            {
                currentTimePoint = 0;
                printTimers();
                //vkQueueWaitIdle(computeQueue);
                //debugSort();
            }
        }
    }

    uint fusionneur(Swap *tab, uint start, uint taille, uint pos){
        for(int i = taille-1; i >= 0; i -= 1)
        {   
            if(pos + 2*taille - i - 1 < opt.nb_particles)
            {
                if(tab != nullptr){
                    tab[start + taille - i - 1].a = pos + i;
                    tab[start + taille - i - 1].b = pos + 2*taille - i - 1;
                }
            }
            else{
                return(taille - i - 1);
            }
        }
        return(taille);
    }

    uint decaleur(Swap *tab, uint start, uint taille, uint pos){
        for(uint i = 0; i < taille; i ++)
        {
            if(pos+i < opt.nb_particles && pos+i+taille < opt.nb_particles)
            {
                if(tab != nullptr){
                    tab[start + i].a = pos+i;
                    tab[start + i].b = pos+i+taille;
                }
            } 
            else{
                return(i);
            }
        }
        return(taille);
    }

    uint passe(Swap *tab, SortBufferObject *taille_swaps, uint taille, uint& nb_stages, uint start_swaps){
        uint pos = 0;
        uint start = start_swaps;
        //printf("fusionneur %u %u %u \n", start, taille, pos);
        uint taille_fus = fusionneur(tab, start, taille, 0);
        //printf(" -> t = %d\n",taille_fus);

        if(taille_swaps != nullptr)
            taille_swaps[nb_stages].begin_swaps = start;

        start += taille_fus;
        pos += 2*taille;
        while(pos < opt.nb_particles)
        {
            //printf("fusionneur %u %u %u \n", start, taille, pos);
            uint t = fusionneur(tab, start, taille, pos);
            //printf(" -> t = %d\n",t);
            if(t != taille)
            {
                //printf("faill de la fusion ! \n");
                taille_fus += t; 
                start += t;
                break;
            }
            else{
                //printf("%d == %d \n", t,taille);
                pos += 2*taille;
                start += taille;
                taille_fus += taille;
            }
        }
        if(taille_swaps != nullptr)
            taille_swaps[nb_stages].nb_swaps = taille_fus;
        nb_stages += 1;
        for(uint j = taille/2; j > 0; j = j/2)
        {
            if(taille_swaps != nullptr)
                taille_swaps[nb_stages].begin_swaps = start;
            uint d_size = 0;
            uint p = 0;
            while(true)
            {
                //printf("decaleur  %u %u %u \n", start, j, p);
                uint t = decaleur(tab, start, j, p);
                //printf(" -> t = %d\n",t);
                if(t != j)
                {
                    start += t;
                    d_size += t;
                    break;
                }
                else 
                {
                    start += j;
                    d_size += j;
                    p += 2*j;
                }

            }
            if(taille_swaps != nullptr)
                taille_swaps[nb_stages].nb_swaps = d_size;
            taille_fus += d_size;
            nb_stages += 1;
        }
        return(taille_fus);
    }

    uint fullSort(Swap *tab, SortBufferObject *taille_swaps, uint& nb_stages){
        nb_stages = 0;
        uint nb_swaps = 0;
        for(int i = 1; i < opt.nb_particles; i *= 2)
        {
            //printf("\npasse %d %d %d \n\n",nb_swaps, i, nb_stages);
            nb_swaps += passe(tab, taille_swaps, i, nb_stages, nb_swaps);
            //printf("\nresultat passe -> %d \n\n",nb_swaps);
        }
        return(nb_swaps);
    }

    void setSortSwaps(){

        uint nb_swaps = fullSort(nullptr, nullptr, nb_sort_stages);
        taille_swaps.resize(nb_sort_stages);
        swaps.resize(nb_swaps);
        printf("calcul taille sort finie ! nbswaps = %d, nb_sort_stages = %d \n",nb_swaps,nb_sort_stages);
        fullSort(swaps.data() ,taille_swaps.data() , nb_sort_stages);
        printf("gen du sort finie ! \n");
        /*printf("|| bitonic sort swaps : \n");
        int j = 0;
        for(int i = 0; i < nb_swaps; i ++)
        {
            if(j < nb_sort_stages && i == taille_swaps[j].begin_swaps)
            {
                printf("|| nouveeau stage ! début à %d, taille %d \n",taille_swaps[j].begin_swaps, 
                    taille_swaps[j].nb_swaps);
                j += 1;
            }
            printf("|| %d <-> %d \n", swaps[i].a, swaps[i].b);

        }*/
    }

    void debugSort(){
        memcpy(chunks.data(),chunkBuffersMapped[currentFrame], sizeof(chunks[1])*chunks.size());
        memcpy(particlesIndex.data(),particlesIndexBuffersMapped[currentFrame], 
                sizeof(particlesIndex[1])*particlesIndex.size());
        memcpy(chunksIndex.data(),chunkIndexBuffersMapped[currentFrame], sizeof(chunksIndex[1])*chunksIndex.size());
        printf("\n");
        for(int i = 0; i < particlesIndex.size(); i ++)
            printf(" %d", particlesIndex[i]);
        printf("\n");
        for(int i = 0; i < chunks.size(); i ++)
            printf(" %d", chunks[i]);
        printf("\n");
        for(int i = 0; i < chunks.size(); i ++)
            printf(" %d[%d-%d]",chunks[i], chunksIndex[chunks[i]].start,chunksIndex[chunks[i]].end);
        printf("\n");
        for(int i = 0; i < opt.nb_chunks; i ++)
            if(!(chunksIndex[i].start == 0 && chunksIndex[i].end == 0))
                printf(" %d[%d-%d]",i, chunksIndex[i].start,chunksIndex[i].end);
        printf("\n\n");
    }

    void initSubmitInfo(){
        computeSubmitInfo.resize(MAX_FRAMES_IN_FLIGHT);
        for(int frame = 0; frame < MAX_FRAMES_IN_FLIGHT; frame++)
        {
            //ASSIGN
            computeSubmitInfo[frame].resize(nb_sort_stages+3);
            computeSubmitInfo[frame][0] = {};
            computeSubmitInfo[frame][0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

            computeSubmitInfo[frame][0].commandBufferCount = 1;
            computeSubmitInfo[frame][0].pCommandBuffers = &assignCommandBuffers[frame];
            computeSubmitInfo[frame][0].signalSemaphoreCount = 1;
            computeSubmitInfo[frame][0].pSignalSemaphores = 
            &sortStageFinishedSemaphores[frame*(nb_sort_stages + 1)];

            //SWAPS SORT
            for(int i = 0; i < nb_sort_stages; i++)
            {
                computeSubmitInfo[frame][i+1] = {};
                computeSubmitInfo[frame][i+1].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

                computeSubmitInfo[frame][i+1].waitSemaphoreCount = 1;
                computeSubmitInfo[frame][i+1].pWaitSemaphores = 
                &sortStageFinishedSemaphores[frame*(nb_sort_stages + 1) + i];
                computeSubmitInfo[frame][i+1].pWaitDstStageMask = waitStagesGeneric;

                computeSubmitInfo[frame][i+1].commandBufferCount = 1;
                computeSubmitInfo[frame][i+1].pCommandBuffers = &sortCommandBuffers[frame*nb_sort_stages + i];
                computeSubmitInfo[frame][i+1].signalSemaphoreCount = 1;
                computeSubmitInfo[frame][i+1].pSignalSemaphores = 
                &sortStageFinishedSemaphores[frame*(nb_sort_stages + 1) + i + 1];
            }
            //STATEND
            computeSubmitInfo[frame][nb_sort_stages+1] = {};
            computeSubmitInfo[frame][nb_sort_stages+1].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

            computeSubmitInfo[frame][nb_sort_stages+1].waitSemaphoreCount = 1;
            computeSubmitInfo[frame][nb_sort_stages+1].pWaitSemaphores = 
            &sortStageFinishedSemaphores[frame*(nb_sort_stages + 1) + nb_sort_stages];
            computeSubmitInfo[frame][nb_sort_stages+1].pWaitDstStageMask = waitStagesGeneric;

            computeSubmitInfo[frame][nb_sort_stages+1].commandBufferCount = 1;
            computeSubmitInfo[frame][nb_sort_stages+1].pCommandBuffers = &statendCommandBuffers[frame];
            computeSubmitInfo[frame][nb_sort_stages+1].signalSemaphoreCount = 1;
            computeSubmitInfo[frame][nb_sort_stages+1].pSignalSemaphores = &statendFinishedSemaphores[frame];

            //COMPUTE
            computeSubmitInfo[frame][nb_sort_stages+2] = {};
            computeSubmitInfo[frame][nb_sort_stages+2].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            computeSubmitInfo[frame][nb_sort_stages+2].waitSemaphoreCount = 1;
            computeSubmitInfo[frame][nb_sort_stages+2].pWaitSemaphores = &statendFinishedSemaphores[frame];
            computeSubmitInfo[frame][nb_sort_stages+2].pWaitDstStageMask = waitStagesGeneric;

            computeSubmitInfo[frame][nb_sort_stages+2].commandBufferCount = 1;
            computeSubmitInfo[frame][nb_sort_stages+2].pCommandBuffers = &computeCommandBuffers[frame];
            computeSubmitInfo[frame][nb_sort_stages+2].signalSemaphoreCount = 1;
            computeSubmitInfo[frame][nb_sort_stages+2].pSignalSemaphores = &computeFinishedSemaphores[frame];

        }
    }

    void initVulkan() {
        //printf("nb triangles = %d \n",indices.size()/3);
        setParticuleNumber(2*COMPUTE_STEP);
        initChunkIndex();
        setVertices();
        setSortSwaps();

        createInstance();
        createSurface();

        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRaymarchingImages();
        createTextureSampler();

        createRenderPass();

        createDescriptorSetLayout();
        createSortingDescriptorSetLayout();

        createGraphicsPipeline();
        createRaymarchingPipeline();
        createComputePipeline(computePipelineLayout,computePipeline, "shaders/comp.spv");
        createComputePipeline(assignPipelineLayout,assignPipeline, "shaders/assign.spv");
        createSortPipeline();
        createComputePipeline(statendPipelineLayout,statendPipeline, "shaders/statend.spv");

        createCommandPool();


        createDepthResources();
        createFramebuffers();
        createRaymarchingDepthResources();
        createRaymarchingFramebuffers();

        createChunkIndexBuffer();
        createParticlesIndexBuffer();
        createChunkBuffer();

        createVertexBuffer();
        createIndexBuffer();
        createParticleBuffer();
        createUniformBuffers();
        createSommetBuffer();
        createSwapBuffer();
        createSortBuffers();

        //createTriangleBuffer();

        createDescriptorPool();
        createSortingDescriptorPool();
        createDescriptorSets();
        createSortingDescriptorSets();

        createCommandBuffers(commandBuffers, MAX_FRAMES_IN_FLIGHT);
        createCommandBuffers(raymarchingCommandBuffers, MAX_FRAMES_IN_FLIGHT);
        createCommandBuffers(computeCommandBuffers, MAX_FRAMES_IN_FLIGHT);
        createCommandBuffers(assignCommandBuffers, MAX_FRAMES_IN_FLIGHT);
        createCommandBuffers(sortCommandBuffers, MAX_FRAMES_IN_FLIGHT*nb_sort_stages);
        createCommandBuffers(statendCommandBuffers, MAX_FRAMES_IN_FLIGHT);

        CreateQueryPool();
        initTimers();

        recordStatendCommandBuffers();
        recordSortCommandBuffers();
        recordAssignCommandBuffers();
        recordComputeCommandBuffers();

        createSyncObjects();
        initSubmitInfo();

        initCamera();
        printf("|   dt  |-| assign || sort || statend || comp |-| rend |\n");
    }

    void drawFrameV2() {


        //turn += 1;
        if(currentFrame == 0){ 
            //printf("fence to get is currently in a %d state \n", 
                //vkGetFenceStatus(device,computeInFlightFences[MAX_FRAMES_IN_FLIGHT-1]));
            vkWaitForFences(device, 1, &computeInFlightFences[MAX_FRAMES_IN_FLIGHT-1], VK_TRUE, UINT64_MAX);

            updateUniformBuffer(currentFrame);

            vkResetFences(device, 1, &computeInFlightFences[MAX_FRAMES_IN_FLIGHT-1]);
        }
        else{
            //printf("fence to get is currently in a %d state \n", 
                //vkGetFenceStatus(device,computeInFlightFences[currentFrame-1]));
            vkWaitForFences(device, 1, &computeInFlightFences[currentFrame-1], VK_TRUE, UINT64_MAX);

            updateUniformBuffer(currentFrame);

            vkResetFences(device, 1, &computeInFlightFences[currentFrame-1]);            
        }
        //printf("on passe le submit ? \n");

        VkResult error1 = vkQueueSubmit(computeQueue, nb_sort_stages+3, computeSubmitInfo[currentFrame].data(), 
            computeInFlightFences[currentFrame]);
        if (error1 != VK_SUCCESS) {
            printf("ERREUR  = %d \n",error1);
            throw std::runtime_error("failed to submit compute command buffer!");
        };

        // Graphics submission
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        //printf("device lost ? = %d \n", vkGetFenceStatus(device,computeInFlightFences[MAX_FRAMES_IN_FLIGHT-1]));
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, 
            swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        VkPipelineStageFlags waitStagesPreRender[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT};

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &computeFinishedSemaphores[currentFrame];
        submitInfo.pWaitDstStageMask = waitStagesPreRender;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &preRaymarchingFinishedSemaphores[currentFrame];
        //printf("device lost ? = %d \n", vkGetFenceStatus(device,computeInFlightFences[MAX_FRAMES_IN_FLIGHT-1]));
        VkResult error = vkQueueSubmit(graphicsQueue, 1, &submitInfo, nullptr);
        if (error != VK_SUCCESS) {
            printf("ERREUR  = %d\n",error);
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        vkResetCommandBuffer(raymarchingCommandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
        recordRaymarchingCommandBuffer(raymarchingCommandBuffers[currentFrame], imageIndex);

        VkSemaphore waitSemaphores[] = 
            { preRaymarchingFinishedSemaphores[currentFrame], imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = 
            { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        submitInfo.waitSemaphoreCount = 2;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &raymarchingCommandBuffers[currentFrame];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderFinishedSemaphores[currentFrame];
        //printf("device lost ? = %d \n", vkGetFenceStatus(device,computeInFlightFences[MAX_FRAMES_IN_FLIGHT-1]));
        VkResult error2 = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);
        if (error2 != VK_SUCCESS) {
            printf("ERREUR  = %d\n",error2);
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinishedSemaphores[currentFrame];

        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;

        presentInfo.pImageIndices = &imageIndex;
        //printf("device lost ? = %d \n", vkGetFenceStatus(device,computeInFlightFences[MAX_FRAMES_IN_FLIGHT-1]));
        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }
        if(currentFrame == 0)
            FetchRenderTimeResults(MAX_FRAMES_IN_FLIGHT - 1);
        else 
            FetchRenderTimeResults(currentFrame-1);
        //printf("device lost ? = %d \n", vkGetFenceStatus(device,computeInFlightFences[MAX_FRAMES_IN_FLIGHT-1]));
        auto currentTime = std::chrono::high_resolution_clock::now();
        deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - timeOfLastUpdate).count();
        timeOfLastUpdate = currentTime;
        
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            moveCamera();
            //computeFrame();
            drawFrameV2();
            //debugSort();
        }
        vkDeviceWaitIdle(device);
    }

    void cleanup() {

        vkDestroyQueryPool(device, timeQueryPool, nullptr);

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        vkDestroyBuffer(device, swapBuffer, nullptr);
        vkFreeMemory(device, swapBufferMemory, nullptr);
    
        //vkDestroyBuffer(device, shaderStorageBuffersTriangles, nullptr);
        //vkFreeMemory(device, shaderStorageBuffersTrianglesMemory, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, shaderStorageBuffers[i], nullptr);
            vkFreeMemory(device, shaderStorageBuffersMemory[i], nullptr);

            vkDestroyBuffer(device, shaderStorageBuffersSommets[i], nullptr);
            vkFreeMemory(device, shaderStorageBuffersSommetsMemory[i], nullptr);

            vkDestroyBuffer(device, chunkBuffers[i], nullptr);
            vkFreeMemory(device, chunkBuffersMemory[i], nullptr);

            vkDestroyBuffer(device, chunkIndexBuffers[i], nullptr);
            vkFreeMemory(device, chunkIndexBuffersMemory[i], nullptr);

            vkDestroyBuffer(device, particlesIndexBuffers[i], nullptr);
            vkFreeMemory(device, particlesIndexBuffersMemory[i], nullptr);
        
        }

        for(int i = 0; i < nb_sort_stages; i ++)
        {
            vkDestroyBuffer(device, sortBuffers[i],nullptr);
            vkFreeMemory(device, sortBuffersMemory[i], nullptr);
        }
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
            vkDestroySemaphore(device, computeFinishedSemaphores[i], nullptr); 
            vkDestroySemaphore(device, statendFinishedSemaphores[i], nullptr); 
            vkDestroyFence(device, computeInFlightFences[i], nullptr);
        }
        for(size_t i = 0; i < swapChainImages.size(); i++)
            vkDestroyFence(device, imagesInFlight[i], nullptr);

        for(int i = 0; i < MAX_FRAMES_IN_FLIGHT*(nb_sort_stages+1); i++)
            vkDestroySemaphore(device, sortStageFinishedSemaphores[i], nullptr); 

        vkDestroyCommandPool(device, commandPool, nullptr);

        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipeline(device, computePipeline, nullptr);
        vkDestroyPipeline(device, assignPipeline, nullptr);
        vkDestroyPipeline(device, sortPipeline, nullptr);
        vkDestroyPipeline(device, statendPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
        vkDestroyPipelineLayout(device, assignPipelineLayout, nullptr);
        vkDestroyPipelineLayout(device, sortPipelineLayout, nullptr);
        vkDestroyPipelineLayout(device, statendPipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroyImage(device, depthImage, nullptr);
        vkDestroyImageView(device, depthImageView, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);

        vkDestroySwapchainKHR(device, swapChain, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, uniformBuffers_opt[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory_opt[i], nullptr);
        }

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorPool(device, sortingDescriptorPool, nullptr);

        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, sortingDescriptorSetLayout, nullptr);

        vkDestroyDevice(device, nullptr);

        vkDestroySurfaceKHR(instance, surface, nullptr);

        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }
};

int main() {
    srand(time(0));
    printf("here we go -> %d !\n", (-1) % 10);
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
