import zmq,time,sys                                                                                                                        

ctx =zmq.Context()                                                                                                                         
socket = ctx.socket(zmq.REQ)                                                                                                               
socket.connect('tcp://localhost:9999')                                                                                                     
for i in range(10):                                                                                                                       
    socket.send(str(time.time()).encode())                                                                                                          
    msg = socket.recv()                                                                                                                    
    print("message id",i,"RT",time.time()-float(msg))