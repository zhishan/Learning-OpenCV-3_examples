
function(brainstorm src)
    get_filename_component(Name ${src} NAME)
    message(STATUS "BrainStorm Ideas ${Name}")
    add_executable(${Name} ${src}.cpp)
    set(additional_libs ${ARGN})
    target_link_libraries(${Name}
        ${additional_libs}
        ${OpenCV_LIBS}
    )
endfunction()


