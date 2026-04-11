You are an expert ML Engineer with experience in low level hardware as well as RL algorithms. You are assigned to develop a sim2real framework for the overcooked environment. The model's training pipeline is completed, it now needs to exported to ESP32 hardware. 

The model for the actor is stored in the file /mnt/Stuff/phd_projects/overcooked/python/networks.py named Actor. This actor has been quantized and exported as an .espdl file. Now, in the /mnt/Stuff/phd_projects/overcooked/esp/esp32s3_xiaosense folder, an inference pipeline needs to be built using esp idf and esp dl. The camera code has been setup. After the policy has been run, the actions need to be sent to an Arduino Nano BLE Sense 33 Rev2 board which will handle action execution.

You need to make the inference pipeline using esp-idf. Right now, just focus on doing it in the esp32s3 xiao sense board as the code will be transferable. For model specific information, check networks.py, train.py and utils.py if necessary

Use google to get enough information about espidf and esp dl. For better storage utilize esp32 PSRAM and partition.csv files

Connection between arduino and esp will happen via spi. The arduino's sole purpose is to execute actions given by the policy. The policy will execute actions based observations from its camera. There is the question of how the policy will infer arena layout from camera alone, that is a question I will address later, right now I just need to get the setup up and running. The ESP32 board will not receive any information from the server.

Do not run any python commands
