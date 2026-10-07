cmake_minimum_required(VERSION 3.22)
project(omni_gateway_next LANGUAGES CXX)

include(CTest)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(gateway_core
    src/cobs.cpp
    src/drive_safety.cpp
    src/move.cpp
    src/MU3.cpp
    src/passthrough.cpp
    src/uart.cpp
    src/udp.cpp
)
target_include_directories(gateway_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(gateway_core PRIVATE -Wall -Wextra -Wpedantic -Wno-psabi)

add_executable(omni_gateway_next src/main.cpp)
target_link_libraries(omni_gateway_next PRIVATE gateway_core)

if(BUILD_TESTING)
    add_executable(test_drive_safety tests/test_drive_safety.cpp src/drive_safety.cpp)
    target_include_directories(test_drive_safety PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME drive_safety COMMAND test_drive_safety)

    add_executable(test_omni_velocity tests/test_omni_velocity.cpp src/move.cpp)
    target_include_directories(test_omni_velocity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME omni_velocity COMMAND test_omni_velocity)

    add_executable(test_uart_passthrough
        tests/test_uart_passthrough.cpp
        src/cobs.cpp
        src/passthrough.cpp
        src/uart.cpp
    )
    target_include_directories(test_uart_passthrough PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME uart_passthrough COMMAND test_uart_passthrough)
endif()

if(BUILD_TESTING)
    add_executable(test_mu3_navigation tests/test_mu3_navigation.cpp)
    target_include_directories(test_mu3_navigation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME mu3_navigation COMMAND test_mu3_navigation)
endif()

if(BUILD_TESTING)
    add_executable(test_mu3_udp_rearm tests/test_mu3_udp_rearm.cpp)
    target_link_libraries(test_mu3_udp_rearm PRIVATE gateway_core)
    add_test(NAME mu3_udp_rearm COMMAND test_mu3_udp_rearm)

    add_executable(test_udp_safety tests/test_udp_safety.cpp)
    target_link_libraries(test_udp_safety PRIVATE gateway_core)
    add_test(NAME udp_safety COMMAND test_udp_safety)

    add_executable(test_serial_io tests/test_serial_io.cpp)
    target_link_libraries(test_serial_io PRIVATE gateway_core)
    add_test(NAME serial_io COMMAND test_serial_io)

    # Assertion-based tests must remain active in release/RelWithDebInfo builds.
    foreach(test_target test_drive_safety test_omni_velocity test_uart_passthrough
                        test_mu3_navigation test_mu3_udp_rearm test_udp_safety test_serial_io)
        target_compile_options(${test_target} PRIVATE -UNDEBUG)
    endforeach()

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME gateway_emergency_shutdown
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gateway_shutdown.py
                         $<TARGET_FILE:omni_gateway_next>)
        set_tests_properties(gateway_emergency_shutdown PROPERTIES TIMEOUT 15)
    endif()
endif()
