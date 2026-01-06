#!/usr/bin/env python3
"""
PLY to OBJ Converter with UV Mapping and Texture Baking

This script converts a PLY mesh with vertex colors to an OBJ mesh with 
UV-mapped textures. It uses Blender's Python API and must be run within Blender.

Requirements:
    - Blender 4.0+ (tested with Blender 4.2)

Usage:
    blender --background --python tools/ply_to_obj.py -- input.ply output.obj

    Or from command line with full paths:
    /path/to/blender -b -P tools/ply_to_obj.py -- /path/to/mesh.ply /path/to/output.obj

The script will:
    1. Import the PLY file
    2. Generate UV coordinates using Smart UV Project
    3. Bake vertex colors to a texture
    4. Export as OBJ with MTL and texture files

Output files:
    - output.obj          - The mesh with UV coordinates
    - output.mtl          - Material definition file
    - vertexcol_bake.png  - Baked vertex color texture (4096x4096)
"""

import bpy
import os
import sys


def convert_ply_to_obj(ply_path: str, obj_path: str, texture_resolution: int = 4096):
    """
    Convert PLY mesh to OBJ with UV mapping and texture baking.
    
    Args:
        ply_path: Path to input PLY file
        obj_path: Path to output OBJ file
        texture_resolution: Resolution of baked texture (default 4096)
    """
    out_dir = os.path.dirname(obj_path)
    obj_name = os.path.basename(obj_path)
    tex_name = "vertexcol_bake.png"
    
    # Ensure output directory exists
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Start with empty scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import PLY
    print(f"Importing PLY: {ply_path}")
    try:
        bpy.ops.wm.ply_import(filepath=ply_path)  # Blender 4.0+
    except AttributeError:
        bpy.ops.import_mesh.ply(filepath=ply_path)  # Blender 3.x fallback

    # Get imported object
    try:
        obj = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = obj
    except IndexError:
        print("Error: PLY import failed or file is empty.")
        return False

    # Prepare mesh for UV unwrapping
    print("Generating UV coordinates...")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.remove_doubles()

    # UV unwrap with Smart UV Project
    bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02, scale_to_bounds=True)
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(rotate=True, margin=0.01)

    # Check for overlaps
    print("Checking for UV overlaps...")
    bpy.ops.uv.select_overlap()
    if any(f.select for f in obj.data.polygons):
        print("Warning: UV overlaps detected. Re-packing with larger margin...")
        bpy.ops.uv.pack_islands(rotate=True, margin=0.05, pin=False)
        bpy.ops.uv.select_all(action='DESELECT')
    else:
        print("UV layout is clean.")

    bpy.ops.object.mode_set(mode='OBJECT')

    # Create baking material
    print("Setting up baking material...")
    mat = bpy.data.materials.new("BakeMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new("ShaderNodeOutputMaterial")
    vcol_node = nodes.new("ShaderNodeVertexColor")
    img_node = nodes.new("ShaderNodeTexImage")
    emit_node = nodes.new("ShaderNodeEmission")

    # Create image for baking
    img_node.image = bpy.data.images.new(tex_name, texture_resolution, texture_resolution)
    nodes.active = img_node

    # Link for baking
    links.new(vcol_node.outputs["Color"], emit_node.inputs["Color"])
    links.new(emit_node.outputs["Emission"], out_node.inputs["Surface"])

    obj.data.materials.append(mat)

    # Configure and run bake
    print("Baking vertex colors to texture...")
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'
    scn.cycles.bake_type = 'EMIT'
    scn.render.bake.use_pass_direct = False
    scn.render.bake.use_pass_indirect = False
    scn.render.bake.margin = 16
    scn.render.bake.cage_extrusion = 0.01
    scn.render.bake.max_ray_distance = 0.01

    bpy.ops.object.bake(type='EMIT')
    print("Bake completed.")

    # Save texture
    print("Saving texture...")
    img_path = os.path.join(out_dir, tex_name) if out_dir else tex_name
    img_node.image.filepath_raw = img_path
    img_node.image.file_format = 'PNG'
    img_node.image.save()

    # Rewire material for final export
    bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
    nodes.remove(emit_node)
    links.new(img_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], out_node.inputs["Surface"])

    # Export OBJ
    print(f"Exporting OBJ to: {obj_path}")
    bpy.ops.wm.obj_export(
        filepath=obj_path,
        export_selected_objects=True,
        export_uv=True,
        export_materials=True,
        path_mode='COPY',
        forward_axis='Y',
        up_axis='Z'
    )

    print(f"✓ Conversion finished: {obj_path}")
    return True


def main():
    # Parse command line arguments
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
        if len(argv) < 2:
            print("Usage: blender --background --python ply_to_obj.py -- <input.ply> <output.obj>")
            sys.exit(1)
        ply_path = os.path.abspath(argv[0])
        obj_path = os.path.abspath(argv[1])
    else:
        print("Error: Must provide input and output paths after '--'")
        print("Usage: blender --background --python ply_to_obj.py -- <input.ply> <output.obj>")
        sys.exit(1)

    success = convert_ply_to_obj(ply_path, obj_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

