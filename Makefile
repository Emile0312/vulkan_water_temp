CFLAGS =  -std=c++17 
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi
DEBUGFLAGS = -fsanitize=address -g

VulkanTest: main.cpp
	g++ $(CFLAGS) -o VulkanTest main.cpp $(LDFLAGS)

.PHONY: test clean

Debug:	main.cpp
	g++ $(CFLAGS) -o VulkanTest main.cpp $(LDFLAGS) $(DEBUGFLAGS)

DebugTest: Debug
	./VulkanTest

test: VulkanTest
	./VulkanTest

clean:
	rm -f VulkanTest
