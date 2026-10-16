idf_component_register(SRCS "main.c" "engine.c" "dot_s3.S" INCLUDE_DIRS "."
                       REQUIRES esp_partition esp_timer driver esp_psram vfs spi_flash)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffast-math)
