"""Render SVR results under fixed camera pose with floor (shadows)."""
import bpy, sys, os
from math import pi
import numpy as np

def new_plane(mylocation, mysize, myname):
    """Add a x-y plane to project shadows on."""
    bpy.ops.mesh.primitive_plane_add(
        size=mysize,
        calc_uvs=True,
        enter_editmode=False,
        align='WORLD',
        location=mylocation,
        rotation=(0, 0, 0),
        scale=(0, 0, 0))
    current_name = bpy.context.selected_objects[0].name
    plane = bpy.data.objects[current_name]
    plane.name = myname
    plane.data.name = myname + "_mesh"
    return

def parent_obj_to_camera(b_camera):
    """Fix camera to look at the origin."""
    origin = (0.0, 0.0, 0.0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    return b_empty

# Get CDC repository path
file_dir = os.path.dirname(os.path.abspath(__file__))
current_folder = os.path.dirname(os.path.dirname(file_dir))

# Add point light source
spot_light_data = bpy.data.lights.new(name = 'spot_light', type = 'POINT')
spot_light_data.energy = 500
spot_light_data.diffuse_factor = 1.0
spot_light_data.specular_factor = 0.0
spot_light_data.volume_factor = 5.0
spot_light_data.shadow_soft_size = 0.25
light2 = bpy.data.objects.new(name = 'light_2', object_data = spot_light_data)

# Specify light location and rotations 
light2.location = [1.7867, 0.34837, 2.5011]
light2.rotation_euler = [1.1175,-0.464,2.32]
bpy.context.collection.objects.link(light2)
bpy.context.view_layer.objects.active = light2
bpy.context.scene.view_settings.view_transform = 'Standard'

# Set world color
bpy.data.worlds['World'].node_tree.nodes["Background"].inputs[0].default_value = (1.0, 1.0, 1.0, 1)

# The model folder name
model_names = ['inpaint_050']
for model_name in model_names:
    # Absolute input folder path of in-the-wild SVR results. NOTE: Depend on where the user saves them
    input_folder = current_folder + f"/SVR/result/sofa_sample/{model_name}"
    pcd_folder = "ply_files"
    ply_folder = os.path.join(input_folder, pcd_folder)
    ply_files = os.listdir(ply_folder)

    # Save directory, save the rendered images under the foler "rendered view with shadow"
    save_dir = os.path.join(input_folder, "rendered_view_w_shadow")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for ply in ply_files:
        filename = ply.split('.')[0]

        # Delete the initialised cube or the left-over mesh from last rendering
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        for this_obj in bpy.data.objects:
            if this_obj.type == "MESH":
                this_obj.select_set(True)
                bpy.ops.object.delete(use_global=False, confirm=False)

        # Load new mesh        
        file_loc = os.path.join(ply_folder, ply)
        imported_object = bpy.ops.import_mesh.ply(filepath=file_loc)
        for this_obj in bpy.data.objects:
            if this_obj.type == "MESH":
                this_obj.select_set(True)
                bpy.context.view_layer.objects.active = this_obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.split_normals()

        # Smoothen the mesh polygons
        bpy.ops.object.mode_set(mode='OBJECT')
        for poly in bpy.data.objects[filename].data.polygons:
            poly.use_smooth = True

      
        # Initialise the material for the mesh, we liked cyan with rough texture
        MAT_NAME = "cyan_color"
        bpy.data.materials.new(MAT_NAME)
        material = bpy.data.materials[MAT_NAME]
        material.use_nodes = True
        material.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 1.0
        material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.084, 0.364, 0.448, 1)
        bpy.data.objects[filename].data.materials.append(material)

        # Specify the image resolution and background transparency
        scene = bpy.context.scene
        scene.render.resolution_x = 1024
        scene.render.resolution_y = 1024
        scene.render.resolution_percentage = 100
        bpy.context.scene.render.film_transparent = False

        # Rotate the mesh to fit into blender axis frame
        bpy.context.object.rotation_euler[0] = pi / 2

        # Configure cameras for rendering
        cam = bpy.data.objects['Camera']
        cam_constraint = cam.constraints.new(type='TRACK_TO')
        cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        cam_constraint.up_axis = 'UP_Y'
        bpy.data.cameras['Camera'].clip_end = 500
        bpy.data.cameras['Camera'].lens = 35
        bpy.data.cameras['Camera'].sensor_height = 32
        bpy.data.cameras['Camera'].sensor_width = 32
        b_empty = parent_obj_to_camera(cam)
        cam_constraint.target = b_empty
        # Cmaera location is fixed in this case
        cam.location = (0.99829, - 0.63175, 0.78099)

        # Set a plane to project mesh shadows on
        new_plane((0,0, -bpy.context.object.dimensions.y / 2), 20, "MyFloor")
        
        # Render images and save them to the saving directory
        bpy.context.scene.render.image_settings.color_mode = 'RGB'
        bpy.context.scene.render.filepath = save_dir + f'/{filename}.png'
        bpy.ops.render.render(write_still=True)



            