import gym

from wrappers.cartpole_v0_http_layer_v1 import HttpController, HttpClient, CartPoleV0HttpHandler, HttpMethods

class RestfulCartPoleV0Wrapper(gym.Wrapper):

    def __init__(self,gym_env,default_cart_mime_type, default_pole_mime_type):
        env = gym_env
        super(RestfulCartPoleV0Wrapper, self).__init__(env)
        handler = CartPoleV0HttpHandler(env)
        service = HttpController(handler)
        self.client = HttpClient(service)
        self.client.DEFAULT_CART_MIME_TYPE = default_cart_mime_type
        self.client.DEFAULT_POLE_MIME_TYPE = default_pole_mime_type
    
    def step(self,action):
        return self.client.step(action)
    
    def reset(self, **kwargs):
        return self.client.reset()
