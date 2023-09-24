import os
import sys
import cv2 as cv
import gymnasium
import numpy as np
import sim
import time
import random
import math
import random
from matplotlib import pyplot as plt 
import torch as th   
from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from gymnasium import spaces, error, utils
from gymnasium.utils import seeding
from typing import Optional
#----------------------------------------------------------------------------------------------------------------------------
STEREO_CAMERA_NAME = "Camara" # Nombre del sensor de vision
#----------------------------------------------------------------------------------------------------------------------------
class GymCoppManR(gymnasium.Env):
    metadata = {"render_modes":["human", "rgb_array"], "render_fps":4}
    def __init__(self):      
        # CONEXION AL ENTORNO
        sim.simxFinish(-1) # Cerrar todas las conexiones abiertas
        self.clientID = sim.simxStart('127.0.0.1', 19999, True, True, 3000, 5)
        if self.clientID!=-1:
            print('Conectado al servidor API remoto')
            sim.simxSynchronous(self.clientID,True)
        else:
            print('Conexion no exitosa')
            sys.exit('No se puede conectar')
        # VARIABLES
        self.POS_MIN, self.POS_MAX = [0.5, -0.5, 0.025], [1.5, 0.5, 0.25]   # Area de movimiento de objetivo aleatorio
        self.cameras = [] # Imagen de camara estereo
        self.x, self.y, self.w, self.h = 85, 85, 85, 85   # ROI-Imagen de entrada
        self.image_width, self.image_height = 85, 85   # Ancho y altura espacio de observacion
        self.DoF = 7   # Numero de articulaciones
        self.action_bound = [[-180, 180], [-180, 180], [-180, 180], [-180, 180], [-180, 180], [-180, 180], [-180, 180]]
        self.jointHandlers = []   # Vector para almacenar articulaciones
        self.reward = 0.0   # Recompensa
        self.n_actions = 7  # Salida de espacio de accion
        self.step_counter = 0   # Contador de episodio
        self.i = 0 
        
        self.default_pos = np.asarray([2.5444438e-14,2.5934508e-01,0.0000000e+00,-9.0046700e+01,-1.0882580e-02,8.9987617e+01,-1.0177775e-13])
        self.a_min = -3.14
        self.a_max = 3.14
        # Obtener los manejadores de la escena
        _, self.efector = sim.simxGetObjectHandle(self.clientID, 'FrankaGripper', sim.simx_opmode_blocking)   # EfectorFinal
        _, self.tcp = sim.simxGetObjectHandle(self.clientID, 'Punta', sim.simx_opmode_blocking)   # Punta TCP
        _, self.objetivo = sim.simxGetObjectHandle(self.clientID, 'Esfera', sim.simx_opmode_blocking)   # Esfera
        for i in range(1, 3):
            _, self.cam = sim.simxGetObjectHandle(self.clientID, STEREO_CAMERA_NAME + str(i), sim.simx_opmode_blocking)
            self.cameras.append(self.cam)
        self.stereo_matcher = cv.StereoBM_create(numDisparities=16, blockSize=15) # Disparidad divisible para 16
        # Obtener los manejadores de articulaciones de Franka Emika Panda
        self.name_base = '/Franka/joint'
        self.res, self.joinbase = sim.simxGetObjectHandle(self.clientID, self.name_base, sim.simx_opmode_blocking)
        for i in range(1, 7):
            self.jointName = f'/Franka/link{i+1}'+'_resp/joint'
            self.res, self.jointHandler = sim.simxGetObjectHandle(self.clientID, self.jointName, sim.simx_opmode_blocking)
            if self.res == sim.simx_return_ok:
                self.jointHandlers.append(self.jointHandler)
            else:
                print('No se pudo obtener el manejador de articulación', jointName)
        self.jointHandlers.insert(0, self.joinbase) # Manejadores de las articulaciones Franka Emika Panda
        # print('Manejadores de articulaciones:', jointHandlers)   # Imprimir los manejadores de articulación
        # ESPACIO DE OBSERVACION        
        #self.observation_space = Dict(
         #   {'c_image': spaces.Box(0, 255, [self.image_height, self.image_width, 1], dtype=np.uint8)} # dtype=np.uint8
        #)
        #self.observation_space = spaces.Box(low=0, high=255, shape=(self.image_height, self.image_width, 1), dtype=np.uint8) # imgs
        self.observation_space = spaces.Box(shape=(11+self.DoF,), low=-180, high=180, dtype = 'float32')   # joints + otros
        # ESPACIO DE ACCION
        #self.action_space = spaces.discrete.Discrete(self.n_actions)
        self.action_space = spaces.Box(-1., 1, shape = (self.n_actions,), dtype = 'float32') 
#----------------------------------------------------------------------------------------------------------------------------
#                                                  METODOS GYMNASIUM
#----------------------------------------------------------------------------------------------------------------------------
    def __del__(self): # no cambiar
        self.close()

    def step(self, action):
        self.step_counter += 1
        print('Contador: ', self.step_counter)
        self.trun, self.done = False, False 
        rd, rp, rc, rl, rb = 0.0, 0.0, 0.0, 0.0, 0.0
        # OBTENER ESTADOS
        ex1, ey1, ez1 = self.get_posicion(self.tcp)
        ox1, oy1, oz1 = self.get_posicion(self.objetivo)
        dist_p = np.sqrt((ex1 - ox1) ** 2 + (ey1 - oy1) ** 2 + (ez1 - oz1) ** 2)
        print(f'Distancia previa : {dist_p:.3f} m')
        self.angles = self.get_angles()
        # EJECUTAR ACCION  
        #action_c = action*180/math.pi # Conversion radianes - grados    
        for i in range(self.DoF):
            self.angles[i] += action[i]
            angle = np.clip(self.angles[i], *self.action_bound[i])   # Valores min/max angulos
            self.move_joint(i, angle)
            #action_r = np.clip(action, self.a_min, self.a_max)   
        #self.set_posicion_din(action_r)   # Establecer una accion
        time.sleep(0.1)   # 100 ms
        self.pos_ef = self.get_posicion(self.tcp)   # posicion tcp
        self.pos_obj = self.get_posicion(self.objetivo)   # Posicion esfera objetivo
        #print('Posicion efector final: ', self.pos_ef[0], self.pos_ef[1], self.pos_ef[2])
        #print('Posicion objetivo: ', self.pos_obj[0], self.pos_obj[1],self.pos_obj[2])
        # OBTENER INFORMACION ADICIONAL
        self.info = self._get_info()
        self.obs = self._get_obs()   # Imagenes / estados * 180 / math.pi
        # RECOMPENSAS
        ex2, ey2, ez2 = self.get_posicion(self.tcp)
        ox2, oy2, oz2 = self.get_posicion(self.objetivo)
        dist_a = np.sqrt((ex2 - ox2) ** 2 + (ey2 - oy2) ** 2 + (ez2 - oz2) ** 2)
        rd = -dist_a
        rb = -0.1*np.square(action).sum()
        print(f'Distancia actual: {dist_a:.3f} m')
        print(f'Recompensa Distancia: {rd:.3f}')
        print('Recompensa Costo:', rb)
        if dist_p < dist_a:   # distancia
            rp = dist_p - dist_a
        else:
            rp = np.linalg.norm(dist_p - dist_a)
        print(f'Recompensa diferencia: {rp:.3f}')
        if dist_a <= 0.35:   # llegada
            rl = dist_a
        else:
            rl = 0.0
        print('Recompensa llegada:', rl) 
        if abs(self.pos_ef[2] - self.pos_obj[2]) < 0.05:
            
            self.done = True   
        
        self.reward = rd + rb + rp + rl # Recompensa total
        print('Recompensa Total: ', self.reward)
        #action = th.tensor(action)
        print('--------------------------------------------------------------------')
                     
        return self.obs, self.reward, self.done, self.trun, self.info
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        #step_counter = 0 # Restablecer contador de pasos
        #self.set_posicion(self.objetivo)
         
        self.pos_inicial = self.default_pos.copy() #self.joint_angles()   #self.default_pos.copy()   # Angulos iniciales de articulaciones
        for i in range(self.DoF):
            self.move_joint(i, self.pos_inicial[i])
        #self.set_posicion_din(self.pos_inicial)   # establecer movimiento
        observation = self._get_obs() # Establecer la observacion
        info = self._get_info()
        return observation, info

    def render(self, mode="rgb_array"):
        #imagen = self._get_obs()
        #if mode == "rgb_array":
         #   mlp.imshow(imagen)
          #  mlp.show()
        return #imagen
    
    def close(self): # no cambiar
        sim.simxStopSimulation(self.clientID,sim.simx_opmode_oneshot)
        sim.simxGetPingTime(self.clientID)
        sim.simxFinish(self.clientID)   # Cerrar conexion a CoppeliaSim
#--------------------------------------------------------------------------------------------------------------------------
#                                                FUNCIONES CONTROL/CONFIGURACION
#--------------------------------------------------------------------------------------------------------------------------
    def _get_obs(self):
        imgs = []
        for i in range(2):
            _, res, img = sim.simxGetVisionSensorImage(self.clientID, self.cameras[i], False, sim.simx_opmode_blocking)
            #img = np.asarray(img, dtype=np.uint8)
            img = np.array(img).astype(np.uint8)
            img = np.reshape(img, (res[0],res[1],3))
            img = np.flip(img, axis=0)
            imgs.append(img) # Retornar esta imagen en la funcion
            
        gray_img = cv.cvtColor(imgs[1], cv.COLOR_BGR2GRAY) # escala en gris
        ret, gray_img = cv.threshold(gray_img,127,255,cv.THRESH_BINARY_INV) # Establecer umbral
        rectified_img = cv.equalizeHist(gray_img) # Rectificar la imagen
        kernel = np.ones((5,5), np.uint8) 
        dilated_img = cv.dilate(rectified_img, kernel, iterations=1) # Dilatar la imagen
        thinned_img = cv.ximgproc.thinning(dilated_img) # Adelgazar la imagen
        exp_img = np.expand_dims(thinned_img, axis=2) # expandir imagen procesada
        exp_img = exp_img[self.y:self.y + self.h, self.x:self.x + self.w]

        # Mapa de profundidad
        left = cv.cvtColor(imgs[1], cv.COLOR_RGB2GRAY)
        right = cv.cvtColor(imgs[0], cv.COLOR_RGB2GRAY)
        depth_map = self.stereo_matcher.compute(left,right)   # Imagen de profundidad
        depth_map = depth_map[self.y:self.y + self.h, self.x:self.x + self.w]   # Usar una porcion de la imagen
        depth_map = np.array(depth_map).astype(np.uint8)   # Convertir tipo 16 a 8
        exp_depth = np.expand_dims(depth_map, axis=2) # Expandir imagen (W, H) -> (W, H, 1)
        
        estate = self.estado()     
   
        #return {'c_image': exp_img}
        return estate #exp_depth
    
    def estado(self):
        '''
        Construir un vector de estados para DNN
            Retorno
            vector: [posicion efector final, angulos, distancia efector-objetivo]
        '''
        angulos = self.get_angles()   # Angulos de articulaciones
        distancia = self.distance_to_goal()   # Distancia previa
        pos_efector = self.get_posicion(self.tcp)   # Posicion tcp
        pos_objetivo = self.get_posicion(self.objetivo)   # Posicion objetivo
        ori_efector = self.get_orientacion(self.tcp)   # Orientacion objetivo
        
        sin_cos = []
        for a in angulos:
            sin_cos.append(np.sin(a))
            sin_cos.append(np.cos(a))
        s = np.concatenate([pos_efector, sin_cos], axis=0)
        #vec_est = np.concatenate((s, pos_objetivo))
        return np.insert(s, 17, distancia)   # Vector 18 elementos
    
    def joint_angles(self): # obtener los angulos de las articulaciones
        self.angles = []
        for i in range(self.DoF):
            _, self.angle = sim.simxGetJointPosition(self.clientID, self.jointHandlers[i], sim.simx_opmode_blocking)
            self.angles.append(self.angle* 180 / math.pi) # de rad a grad
            #self.angles.append(self.angle) # rad
        return np.array(self.angles, dtype=np.float32) #self.angles
        
    def get_angles(self):
        angles = []
        for i in range(self.DoF):
            code, angle = sim.simxGetJointPosition(self.clientID, self.jointHandlers[i], sim.simx_opmode_oneshot)
            angles.append(angle * 180 / math.pi)
        return np.array(angles, dtype=np.float32) 
    
    def move_joint(self, num, value):
        '''
        Establecer una posicion a cada articulacion del robot.
        '''
        # in radian
        code = sim.simxSetJointTargetPosition(self.clientID, self.jointHandlers[num], value * math.pi / 180, sim.simx_opmode_blocking)
    
    def set_posicion_din(self, coords): # Establecer posicion del robot
        """
            Parametros
            ----------
            coords : ndarray
                     Coordenadas generalizadas
        """
        sim.simxPauseCommunication(self.clientID, 1)
        for i in range(self.DoF):
            sim.simxSetJointTargetPosition(self.clientID, self.jointHandlers[i], coords[i], sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.clientID, 0)     
        
    def set_position_est(self, coords): # Establecer posicion (escena estatica)
        """
            Parametros
            ----------
            coords : ndarray
                     Coordenadas generalizadas
        """
        sim.simxPauseCommunication(self.clientID, 1)
        for i in range(self.DoF):
            sim.simxSetJointPosition(self.clientID, self.jointHandlers[i], coords[i], sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.clientID, 0)
        
    def get_posicion(self, handle):
        """ return: An array containing the (X,Y,Z,Qx,Qy,Qz,Qw) pose of
            the object.
        """
        _, position = sim.simxGetObjectPosition(self.clientID, handle, -1, sim.simx_opmode_blocking)
        _, quaternion = sim.simxGetObjectQuaternion(self.clientID, handle, -1, sim.simx_opmode_blocking)
        return np.array(position, dtype=np.float32)   #np.r_[position, quaternion]
    
    def set_posicion(self, handle):
        posi = list(np.random.uniform(self.POS_MIN, self.POS_MAX))
        sim.simxSetObjectPosition(self.clientID, handle, -1, posi, sim.simx_opmode_oneshot)
    
    def get_orientacion(self, handle):
        _, orientacion = sim.simxGetObjectOrientation(self.clientID, handle, -1, sim.simx_opmode_oneshot)
        return np.array(orientacion, dtype=np.float32)   # angulos Euler

    def distance_to_goal(self):   
        '''
        Calcular la distancia entre las posiciones del gripper y esfera.
            Retorno
            Distancia en metros.
        '''
        _, self.suelda_pos = sim.simxGetObjectPosition(self.clientID, self.tcp, self.objetivo, sim.simx_opmode_blocking)
        _, self.objetivo_pos = sim.simxGetObjectPosition(self.clientID, self.objetivo, self.tcp, sim.simx_opmode_blocking)
        self.distancia = np.linalg.norm(np.array(self.suelda_pos) - np.array(self.objetivo_pos))
        return np.array(self.distancia, dtype=np.float32) # self.distancia
    
    def tocar(self): # Detectar si hay colision entre el efector y la trayectoria
        _, self.colision = sim.simxCheckCollision(self.clientID, self.tcp, self.objetivo, sim.simx_opmode_blocking)
        self.c = 0.0
        self.colision = False
        if self.colision == True:
            print('Colision!')
            self.c = 10.0
        else:
            print('No hay colision!')
            self.c = 0.0
        return self.c
        
    def _get_info(self): # verificar
        self.distancia = 0.0
        self.distancia = self.distance_to_goal()
        return {
            'info' : self.distancia
        }   
