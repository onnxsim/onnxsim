#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ci.pApplicationInfo = &ai;
  VkInstance inst; if (vkCreateInstance(&ci, nullptr, &inst)) { puts("no instance"); return 1; }
  uint32_t n = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &n, &pd);
  VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd, &p); printf("%s api %u.%u.%u driver %u\n", p.deviceName, VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion), VK_VERSION_PATCH(p.apiVersion), p.driverVersion);
  uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, nullptr); std::vector<VkExtensionProperties> ex(ne); vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, ex.data());
  for (auto& e : ex) if (strstr(e.extensionName, "dot") || strstr(e.extensionName, "8bit") || strstr(e.extensionName, "float16_int8") || strstr(e.extensionName, "shader_subgroup") || strstr(e.extensionName, "cooperative") || strstr(e.extensionName, "storage_buffer_storage_class") || strstr(e.extensionName, "16bit")) printf("ext %s v%u\n", e.extensionName, e.specVersion);
  VkPhysicalDeviceShaderIntegerDotProductFeatures dp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.pNext = &dp;
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f2.pNext = &f12; vkGetPhysicalDeviceFeatures2(pd, &f2);
  printf("shaderInt8=%u storageBuffer8BitAccess=%u uniformAndStorageBuffer8BitAccess=%u shaderFloat16=%u storageBuffer16BitAccess=%u shaderIntegerDotProduct=%u\n", f12.shaderInt8, f12.storageBuffer8BitAccess, f12.uniformAndStorageBuffer8BitAccess, f12.shaderFloat16, 0u, dp.shaderIntegerDotProduct);
  VkPhysicalDeviceShaderIntegerDotProductProperties dpp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES};
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; p2.pNext = &dpp; vkGetPhysicalDeviceProperties2(pd, &p2);
  printf("dot product accel: 8bit signed packed=%u, 4x8 signed=%u, acc-sat 8bit signed packed=%u\n", dpp.integerDotProduct4x8BitPackedSignedAccelerated, dpp.integerDotProduct8BitSignedAccelerated, dpp.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated);
  return 0;
}
