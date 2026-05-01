-- Setup the extension.
local ext = get_current_extension_info()

project_ext(ext)

-- Link folders that should be packaged with the extension.
-- Source layout uses python/src/ but it is deployed as the `impl` submodule
-- to mirror the upstream isaacsim.physics.newton convention.
repo_build.prebuild_link {
    { "data", ext.target_dir.."/data" },
    { "docs", ext.target_dir.."/docs" },
    { "python/src", ext.target_dir.."/minsim/physics/xewton/impl" },
    { "python/tests", ext.target_dir.."/minsim/physics/xewton/tests" },
}

-- Copy the main __init__.py to maintain the module root
repo_build.prebuild_copy {
    { "python/__init__.py", ext.target_dir.."/minsim/physics/xewton/__init__.py" },
}
