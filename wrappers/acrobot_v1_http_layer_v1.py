import enum
import json

#Acrobotv1-OpenAPIv1.json
class HttpController:

    def __init__(self, handler):
        self.handler = handler
    
    def get(self, path, accept=None):
        if path == "/joints":
            return self.handler.getJoints(accept)
        elif path == "/joints/0":
            return self.handler.getJoint(0,accept)
        elif path == "/joints/1":
            return self.handler.getJoint(1,accept)
        elif path == "/env/reward":
            return self.handler.getLastReward(accept)
        elif path == "/env/terminal":
            return self.handler.getTerminal(accept)
        else:
            return self.notFound()
    
    def put(self, path, payload, accept=None):
        if path == "/joints/0":
            return self.methodNotAllowed()
        elif path == "/joints/1":
            return self.handler.setTorque(payload,accept)
        else:
            return self.notFound()

    def post(self, path, payload, accept=None):
        if path == "/env/render":
            return self.handler.render(payload, accept)
        elif path == "/env/reset":
            return self.handler.reset(payload, accept)
        else:
            return self.notFound()

    def delete(self, path, accept=None):
        return self.notFound()

    def methodNotAllowed(self):
        return 405, "This method is not allowed"
    
    def notFound(self):
        return 404, "Resource not found"

    def process(self, method, path, payload, accept= None):
        if method == HttpMethods.GET:
            return self.get(path, accept)
        elif method == HttpMethods.PUT:
            return self.put(path, payload, accept)
        elif method == HttpMethods.POST:
            return self.post(path, payload, accept)
        elif method == HttpMethods.DELETE:
            return self.delete(path, accept)
        else:
            return self.methodNotAllowed()

class HttpMethods (enum.Enum):
    GET = 1
    PUT = 2
    POST = 3
    DELETE = 4    

class AcrobotV1HttpHandlerV1:

    def __init__(self, env):
        self.env = env

        self.MIME_TYPE_JOINT_JSON = "application/vnd.acrobot.joint+json"
        self.MIME_TYPE_JOINTS_JSON = "application/vnd.acrobot.joints+json"
        self.MIME_TYPE_JOINT_XML = "application/vnd.acrobot.joint+xml"
        self.MIME_TYPE_JOINTS_XML = "application/vnd.acrobot.joints+xml"
        self.MIME_TYPE_JOINT_PLAIN_V1 = "text/vnd.acrobot.joint.v1+plain"
        self.MIME_TYPE_JOINTS_PLAIN_V1 = "text/vnd.acrobot.joints.v1+plain"
    
    def getJoints(self, accept):
        #prepare response payload:
        if accept == None or accept == self.MIME_TYPE_JOINTS_JSON:
            #JSON:
            _, joint0 = self.getJoint(0, accept=self.MIME_TYPE_JOINT_JSON)
            _, joint1 = self.getJoint(1, accept=self.MIME_TYPE_JOINT_JSON)
            responsePayload = "{\"joints\":["+joint0+", "+joint1+"]}"
            return 200, responsePayload
        
        elif accept == self.MIME_TYPE_JOINTS_XML:
            #XML:
            _, joint0 = self.getJoint(0, accept=self.MIME_TYPE_JOINT_XML)
            _, joint1 = self.getJoint(1, accept=self.MIME_TYPE_JOINT_XML)
            responsePayload = "<joints>"+joint0+joint1+"</ joints>"
            return 200, responsePayload
        
        elif accept == self.MIME_TYPE_JOINTS_PLAIN_V1:
            _, joint0 = self.getJoint(0, accept=self.MIME_TYPE_JOINT_PLAIN_V1)
            _, joint1 = self.getJoint(1, accept=self.MIME_TYPE_JOINT_PLAIN_V1)
            return 200, joint0+" "+joint1
        else:
            return 406, "The requested content-type is not acceptable for the requested resource" 
    
    def getJoint(self, id, accept):
        if id == 0:
            torque = "idle"
        elif id == 1:
            torque = self.lastTorque
        else:
            return self.notFound()

        #prepare response payload:
        if accept== None or accept == self.MIME_TYPE_JOINT_JSON:
            #JSON:
            responsePayload = {
                "id": id,
                "torque": str(torque),
                "cosine":self.lastObservation[0+(id*2)],
                "sine":self.lastObservation[1+(id*2)],
                "angular velocity":self.lastObservation[4+id]
            }
            encodedJson = json.dumps(responsePayload)
            return 200, encodedJson

        elif accept == self.MIME_TYPE_JOINT_XML:
            #XML:
            return 200, "<joint><id>"+str(id)+"</ id><torque>"+str(torque)+"</ torque><cosine>"+str(self.lastObservation[0+(id*2)])+"</ cosine><sine>"+str(self.lastObservation[1+(id*2)])+"</ sine><angular velocity>"+str(self.lastObservation[4+id])+"</ angular velocity></ joint>"
        elif accept == self.MIME_TYPE_JOINT_PLAIN_V1:
            #TEXT PLAIN V1:
            return 200, str(id)+" "+str(torque)+" "+str(self.lastObservation[0+(id*2)])+" "+str(self.lastObservation[1+(id*2)])+" "+str(self.lastObservation[4+id])
        else:
            return 406, "The requested content-type is not acceptable for the requested resource" 

    def setTorque(self, payload, accept):
        try:
            #parse and interpret request payload:
            decodedJson = json.loads(payload)
            torque = decodedJson["torque"]
            if torque == "left":
                self.lastTorque = "left"
                self.lastObservation, self.lastReward, self.terminal, _ = self.env.step(1)
            elif torque == "right":
                self.lastTorque = "right"
                self.lastObservation, self.lastReward, self.terminal, _ = self.env.step(2)
            elif torque == "idle":
                self.lastTorque = "idle"
                self.lastObservation, self.lastReward, self.terminal, _ = self.env.step(0)
            else:
                return 400, "Bad Request"
            
            #prepare and return response
            return self.getJoint(1,accept)

        except json.JSONDecodeError:
            return 400, "Bad Request"
    
    def render(self, payload, accept):
        self.env.render()
        return 200, "OK"
    
    def reset(self, payload, accept):
        self.lastObservation = self.env.reset()
        self.lastTorque = "idle"
        return self.getJoints(accept)
    
    def getLastReward(self, accept):
        return 200, self.lastReward
    
    def getTerminal(self, accept):
        return 200, self.terminal

class HttpClient:

    def __init__(self, remoteService):
        self.remoteService = remoteService
        self.DEFAULT_JOINT_MIME_TYPE = None
        self.DEFAULT_JOINTS_MIME_TYPE = None

    def step(self, action):
        requestPayload = ""
        if action == 1:
            requestPayload = {
                "torque":"left"
            }
        elif action == -1:
            requestPayload = {
                "torque":"right"
            }
        elif action == 2:
            requestPayload = {
                "torque":"right"
            }
        else:
            requestPayload = {
                "torque":"idle"
            }
        self.remoteService.process(HttpMethods.PUT,"/joints/1",json.dumps(requestPayload),self.DEFAULT_JOINT_MIME_TYPE)
        _, state = self.remoteService.process(HttpMethods.GET,"/joints",None,self.DEFAULT_JOINTS_MIME_TYPE)
        _, reward = self.remoteService.process(HttpMethods.GET,"/env/reward",None)
        _, terminal = self.remoteService.process(HttpMethods.GET,"/env/terminal",None)
        return state, reward, terminal, {}
    
    def reset(self):
        _, temp = self.remoteService.process(HttpMethods.POST,"/env/reset",None,self.DEFAULT_JOINTS_MIME_TYPE)
        return temp

    
    def render(self):
        self.remoteService.process(HttpMethods.POST,"/env/render",None)