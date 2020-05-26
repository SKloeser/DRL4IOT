import enum
import json


# OpenAPIv1.json
class HttpController:

    def __init__(self, handler):
        self.Handler = handler

    def get(self, path, accept=None):
        if path == "/cart":
            return self.Handler.getCart(accept)
        elif path == "/cart/pole":
            return self.Handler.getPole(accept)
        elif path == "/env/reward":
            return self.Handler.getLastReward(accept)
        elif path == "/env/terminal":
            return self.Handler.getTerminal(accept)
        else:
            return self.notFound()

    def put(self, path, payload, accept=None):
        if path == "/cart":
            return self.Handler.setCart(payload, accept)
        else:
            return self.notFound()

    def post(self, path, payload, accept=None):
        if path == "/env/render":
            return self.Handler.render(payload, accept)
        elif path == "/env/reset":
            return self.Handler.reset(payload, accept)
        else:
            return self.notFound()

    def delete(self, path, accept=None):
        return self.notFound()

    def methodNotAllowed(self):
        return 405, "This method is not allowed"

    def notFound(self):
        return 404, "Resource not found"

    def process(self, method, path, payload, accept=None):
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


class HttpMethods(enum.Enum):
    GET = 1
    PUT = 2
    POST = 3
    DELETE = 4


class CartPoleV0HttpHandler:

    def __init__(self, env):
        self.Env = env

        self.MIME_TYPE_POLE_JSON = "application/vnd.cartpole.pole+json"
        self.MIME_TYPE_CART_JSON = "application/vnd.cartpole.cart+json"
        self.MIME_TYPE_POLE_PLAIN_V1 = "text/vnd.cartpole.pole.v1+plain"
        self.MIME_TYPE_CART_PLAIN_V1 = "text/vnd.cartpole.cart.v1+plain"
        self.MIME_TYPE_POLE_XML = "application/vnd.cartpole.pole+XML"
        self.MIME_TYPE_CART_XML = "application/vnd.cartpole.cart+XML"

    def getCart(self, accept):
        # prepare response payload:
        if accept == None or accept == self.MIME_TYPE_CART_JSON:
            # JSON:
            responsePayload = {
                "position": self.LastObservation[0],
                "velocity": self.LastObservation[1],
                "direction": self.LastDirection
            }
            encodedJson = json.dumps(responsePayload)
            return 200, encodedJson

        elif accept == self.MIME_TYPE_CART_XML:
            # XML:
            return 200, "<cart><position>" + str(self.LastObservation[0]) + "</ position><velocity>" + str(
                self.LastObservation[1]) + "</ velocity><direction>" + str(self.LastDirection) + "</ direction></ cart>"
        elif accept == self.MIME_TYPE_CART_PLAIN_V1:
            # TEXT PLAIN V1:
            return 200, str(self.LastObservation[0]) + " " + str(self.LastObservation[1]) + " " + str(
                self.LastDirection)
        else:
            return 406, "The requested content-type is not acceptable for the requested resource"

    def getPole(self, accept):
        # prepare response payload
        if accept == None or accept == self.MIME_TYPE_POLE_JSON:
            # JSON:
            responsePayload = {
                "angle": self.LastObservation[2],
                "speed": self.LastObservation[3]
            }
            encodedJson = json.dumps(responsePayload)
            return 200, encodedJson

        elif accept == self.MIME_TYPE_POLE_XML:
            # XML:
            return 200, "<pole><angle>" + str(self.LastObservation[2]) + "</ angle><speed>" + str(
                self.LastObservation[3]) + "</ speed></ pole>"
        elif accept == self.MIME_TYPE_POLE_PLAIN_V1:
            # TEXT PLAIN V0:
            return 200, str(self.LastObservation[2]) + " " + str(self.LastObservation[3])
        else:
            return 406, "The requested content-type is not acceptable for the requested resource"

    def setCart(self, payload, accept):
        try:
            # parse and interpret request payload:
            decodedJson = json.loads(payload)
            direction = decodedJson["direction"]
            if direction == "left":
                # step left:
                self.LastDirection = "left"
                self.LastObservation, self.LastReward, self.Terminal, _ = self.Env.step(0)
            elif direction == "right":
                # step right:
                self.LastDirection = "right"
                self.LastObservation, self.LastReward, self.Terminal, _ = self.Env.step(1)
            else:
                return 400, "Bad Request"

            # prepare response payload:
            if accept == None or accept == self.MIME_TYPE_CART_JSON:
                # JSON:
                responsePayload = {
                    "position": self.LastObservation[0],
                    "velocity": self.LastObservation[1],
                    "direction": self.LastDirection
                }
                encodedJson = json.dumps(responsePayload)
                return 200, encodedJson

            elif accept == self.MIME_TYPE_CART_XML:
                # XML:
                return 200, "<cart><position>" + str(self.LastObservation[0]) + "</ position><velocity>" + str(
                    self.LastObservation[1]) + "</ velocity><direction>" + str(
                    self.LastDirection) + "</ direction></ cart>"

            elif accept == self.MIME_TYPE_CART_PLAIN_V1:
                # TEXT PLAIN V0:
                return 200, str(self.LastObservation[0]) + " " + str(self.LastObservation[1]) + " " + str(
                    self.LastDirection)

            else:
                return 406, "The requested content-type is not acceptable for the requested resource"

        except json.JSONDecodeError:
            return 400, "Bad Request"

    def render(self, payload, accept):
        self.Env.render()
        return 200, "OK"

    def reset(self, payload, accept):

        self.LastObservation = self.Env.reset()
        self.LastDirection = "left"

        # prepare response payload:
        if accept == None or accept == self.MIME_TYPE_CART_JSON:
            # JSON:
            responsePayload = {
                "position": self.LastObservation[0],
                "velocity": self.LastObservation[1],
                "direction": self.LastDirection
            }
            encodedJson = json.dumps(responsePayload)
            return 200, encodedJson

        elif accept == self.MIME_TYPE_CART_XML:
            # XML:
            return 200, "<cart><position>" + str(self.LastObservation[0]) + "</ position><velocity>" + str(
                self.LastObservation[1]) + "</ velocity><direction>" + str(self.LastDirection) + "</ direction></ cart>"

        elif accept == self.MIME_TYPE_CART_PLAIN_V1:
            # TEXT PLAIN V0:
            return 200, str(self.LastObservation[0]) + " " + str(self.LastObservation[1]) + " " + str(
                self.LastDirection)

        else:
            return 406, "The requested content-type is not acceptable for the requested resource"

        return 200, encodedJson

    def getLastReward(self, accept):
        return 200, self.LastReward

    def getTerminal(self, accept):
        return 200, self.Terminal


class HttpClient:
    def __init__(self, remoteService):
        self.RemoteService = remoteService
        self.DEFAULT_CART_MIME_TYPE = None
        self.DEFAULT_POLE_MIME_TYPE = None

    def step(self, action):
        requestPayload = ""
        if action == 0:
            requestPayload = {
                "direction": "left"
            }

        else:
            requestPayload = {
                "direction": "right"
            }
        self.RemoteService.process(HttpMethods.PUT, "/cart", json.dumps(requestPayload), self.DEFAULT_CART_MIME_TYPE)

        _, cartState = self.RemoteService.process(HttpMethods.GET, "/cart", None, self.DEFAULT_CART_MIME_TYPE)
        _, poleState = self.RemoteService.process(HttpMethods.GET, "/cart/pole", None, self.DEFAULT_POLE_MIME_TYPE)
        _, reward = self.RemoteService.process(HttpMethods.GET, "/env/reward", None)
        _, terminal = self.RemoteService.process(HttpMethods.GET, "/env/terminal", None)
        return cartState + poleState, reward, terminal, {}

    def reset(self):
        self.RemoteService.process(HttpMethods.POST, "/env/reset", None)
        _, cartState = self.RemoteService.process(HttpMethods.GET, "/cart", None, self.DEFAULT_CART_MIME_TYPE)
        _, poleState = self.RemoteService.process(HttpMethods.GET, "/cart/pole", None, self.DEFAULT_POLE_MIME_TYPE)
        return cartState + poleState

    def render(self):
        self.RemoteService.process(HttpMethods.POST, "/env/render", None)