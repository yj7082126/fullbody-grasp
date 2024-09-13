
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#

import os
from pathlib import Path
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import numpy as np
from PIL import Image
import torch

import pyrender
from pyrender.constants import RenderFlags

from psbody.mesh import Mesh, MeshViewers
from psbody.mesh.sphere import Sphere
from psbody.mesh.colors import name_to_rgb
from psbody.mesh.lines import Lines
from tools.utils import euler
import scenepic as sp

def points_to_spheres(points, radius=0.1, vc=name_to_rgb['blue']):
    spheres = Mesh(v=[], f=[])
    for pidx, center in enumerate(points):
        clr = vc[pidx] if len(vc) > 3 else vc
        spheres.concatenate_mesh(Sphere(center, radius).to_mesh(color=clr))
    return spheres

def cage(length=1,vc=name_to_rgb['black']):
    cage_points = np.array([[-1., -1., -1.],
                            [1., 1., 1.],
                            [1., -1., 1.],
                            [-1., 1., -1.]])
    c = Mesh(v=length * cage_points, f=[], vc=vc)
    return c


def create_video(path, fps=30,name='movie'):
    import os
    import subprocess

    src = os.path.join(path,'%*.png')
    movie_path = os.path.join(path,'%s.mp4'%name)
    i = 0
    while os.path.isfile(movie_path):
        movie_path = os.path.join(path,'%s_%02d.mp4'%(name,i))
        i +=1


    cmd = 'ffmpeg -f image2 -r %d -i %s -b:v 6400k -pix_fmt yuv420p %s' % (fps, src, movie_path)
    subprocess.call(cmd.split(' '))
    while not os.path.exists(movie_path):
        continue


def get_ground(cage_size = 7, grnd_size = 5, axis_size = 1):
    ax_v = np.array([[0., 0., 0.],
                     [1.0, 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]])
    ax_e = [(0, 1), (0, 2), (0, 3)]

    axis_l = Lines(axis_size*ax_v, ax_e, vc=np.eye(4)[:, 1:])

    g_points = np.array([[-.2, 0.0, -.2],
                         [.2, 0.0, .2],
                         [.2, 0.0, -0.2],
                         [-.2, 0.0, .2]])
    g_faces = np.array([[0, 1, 2], [0, 3, 1]])
    grnd_mesh = Mesh(v=grnd_size * g_points, f=g_faces, vc=name_to_rgb['gray'])

    cage_points = np.array([[-.2, .0, -.2],
                            [.2, .2, .2],
                            [.2, 0., 0.2],
                            [-.2, .2, -.2]])
    cage = [Mesh(v=cage_size * cage_points, f=[], vc=name_to_rgb['black'])]
    return grnd_mesh, cage, axis_l

class sp_animation():
    def __init__(self,
                 width = 1600,
                 height = 1600,
                 ):
        super(sp_animation, self).__init__()

        self.scene = sp.Scene()
        self.main = self.scene.create_canvas_3d(width=width, height=height)
        self.colors = sp.Colors

    def meshes_to_sp(self,meshes_list, layer_names):
        sp_meshes = []
        for i, m in enumerate(meshes_list):
            params = {'vertices' : m.v.astype(np.float32),
                      'normals' : m.estimate_vertex_normals().astype(np.float32),
                      'triangles' : m.f,
                      'colors' : m.vc.astype(np.float32)}
            sp_m = self.scene.create_mesh(layer_id = layer_names[i])
            sp_m.add_mesh_with_normals(**params)
            if layer_names[i] == 'ground_mesh':
                sp_m.double_sided=True
            sp_meshes.append(sp_m)
        return sp_meshes

    def add_frame(self,meshes_list_ps, layer_names):

        meshes_list = self.meshes_to_sp(meshes_list_ps, layer_names)
        if not hasattr(self,'focus_point'):
            self.focus_point = meshes_list_ps[1].v.mean(0)
        main_frame = self.main.create_frame(focus_point=self.focus_point)
        for i, m in enumerate(meshes_list):
            main_frame.add_mesh(m)

    def save_animation(self, sp_anim_name):
        self.scene.link_canvas_events(self.main)
        self.scene.save_as_html(sp_anim_name, title=sp_anim_name.split('/')[-1])

class CustomShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            shaders_directory = Path('data/GRAB_origin/shaders')
            self.program = pyrender.shader_program.ShaderProgram(
                shaders_directory / 'mesh.vert', 
                shaders_directory / 'mesh.frag',
                defines=defines
            )
        return self.program
    
    def clear(self):
        self.program.delete()

def easy_render(obj_array, angle=[-90,0,0], z_dist=1.5, 
                a_light=0.4, d_light=3.0, 
                w=512, h=512, bg_color=[0.0,0.0,0.0,1.0], 
                normal_render=False,
                flat_render=False,
                return_depthbuffer=False):
    scene = pyrender.Scene(bg_color=bg_color, ambient_light=a_light, name='scene')

    pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([0.0, 0.0, 0.0], 'xzy')
    camera_pose[:3, 3] = np.array([0.0, 0.0, z_dist])
    cam = pyrender.Node(name = 'camera', camera=pc, matrix=camera_pose)
    scene.add_node(cam)

    if d_light > 0:
        light = pyrender.light.DirectionalLight(color=np.ones(3), intensity=d_light)
        light = pyrender.Node(light=light, matrix=camera_pose)
        scene.add_node(light)

    for obj in obj_array:
        obj.rot_verts(euler(angle, 'xzy'))
        mesh = pyrender.Mesh.from_trimesh(obj)
        scene.add(mesh)

    viewer = pyrender.OffscreenRenderer(w, h)
    
    if flat_render:
        flags = RenderFlags.FLAT
    else:
        flags = RenderFlags.NONE

    if normal_render:
        viewer._renderer._program_cache = CustomShaderCache()
    color, depth_buffer = viewer.render(scene, flags=flags)
    
    depth = depth_buffer.copy()
    mask = depth > 0
    color_image = Image.fromarray(np.concatenate([color, (mask[...,np.newaxis]*255.).astype(np.uint8)], axis=-1))

    if np.sum(mask) > 0:
        min_depth, max_depth = depth[mask].min(), depth[mask].max()
        depth[mask] = (depth[mask] - min_depth) / (max_depth - min_depth)
        depth[mask] = (1.0 - depth[mask] * 0.5)
        depth_image = (depth * 255.).astype(np.uint8).clip(0, 255)
        depth_image = Image.fromarray(depth_image)
    else:
        print("Depth Image Fail")
        depth_image = Image.new("L", (w, h))

    for obj in obj_array:
        obj.rot_verts(euler(angle, 'xzy').T)
    viewer.delete()

    if return_depthbuffer:
        return color_image, depth_image, depth_buffer
    else:
        return color_image, depth_image