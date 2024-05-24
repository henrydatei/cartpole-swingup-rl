import zmq,time,sys

ctx =zmq.Context()
socket = ctx.socket(zmq.REP)
socket.bind('tcp://*:9999')

while True:
    msg = socket.recv()
    print("client->server msg took",time.time()-float(msg))
    socket.send(msg)