# CartPole specific stepper control

The arduino files for the stepper control are adjusted for CartPole.

The velocity can be send via "<v,PUT_A_INTEGER_HERE>". It will accelerate to this speed with a given acceleration. Change this acceleration with "<a,PUT_A_INTEGER_HERE>". You can ask for the current position via "<p,0>" the "payload" will be the position in steps. To set the current position as 0 send "<h,0>".  

See arduino_communication_CARTPOLE.py for a demonstration. In there an episode restart is also shown. This way you can also prevent bumping into your rig. Change maxRevolutions in python and cpp to your need.

Note that the original CartPole problem deals with a force rather than speed change with a constant acceleration!

It might be a good idea to reduce the max speed to ~10000, since the microcontroller might not be able to do more, for the slower UNO 4000 seems to be max (https://forum.arduino.cc/t/maximum-speed-using-accelstepper/1059990). If you need fast movements you can reduce the step size from 12800 to something below.

The tracking algorithm I use is also fast enough for this: https://datashare.tu-dresden.de/s/Jz7GSYk7QAapm2K
