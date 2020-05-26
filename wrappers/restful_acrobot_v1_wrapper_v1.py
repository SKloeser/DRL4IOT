import gym

from wrappers.acrobot_v1_http_layer_v1 import HttpController, HttpClient, AcrobotV1HttpHandlerV1, HttpMethods

class RestfulAcrobotV1WrapperV1(gym.Wrapper):

    def __init__(self, gym_evn, default_joint_mime_type, default_joints_mime_type):
        env = gym_evn
        super(RestfulAcrobotV1WrapperV1,self).__init__(env)
        handler = AcrobotV1HttpHandlerV1(env)
        service = HttpController(handler)
        self.client = HttpClient(service)
        self.client.DEFAULT_JOINT_MIME_TYPE = default_joint_mime_type
        self.client.DEFAULT_JOINTS_MIME_TYPE = default_joints_mime_type
    
    def step(self,action):
        return self.client.step(action)
    
    def reset(self, **kwargs): 
        return self.client.reset()

