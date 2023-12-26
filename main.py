from builtins import range
import matplotlib.pyplot as plt
import numpy as np
from generateXML_test import getXML
from helpers_test import *
import torch
import MalmoPython
import json
import os
import sys
import time
import random
import math


TEST_FLAG = 0

IMAGE_SIZE = 256
NUM_ACTIONS = 5
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-4
TARGET_UPDATE = 10


if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

if __name__ == "__main__":
    with open("data_saved.txt", "w") as f:
        f.write("Distances over time\n")

    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:',e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)
    
    agent = DropperAgent(agent_host)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN((IMAGE_SIZE, IMAGE_SIZE), NUM_ACTIONS).to(device)
    #policy_net.load_state_dict(torch.load("saved_params.wts"))
    target_net = DQN((IMAGE_SIZE, IMAGE_SIZE), NUM_ACTIONS).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), LR)
    memory = ReplayMemory(10000)

    missionXML = getXML()

    my_mission = MalmoPython.MissionSpec(missionXML, True)
    my_mission.requestVideo(IMAGE_SIZE, IMAGE_SIZE)
    my_mission_record = MalmoPython.MissionRecordSpec("data")
    my_mission_record.recordMP4(30, 5000)
    my_mission_record.recordObservations()

    episode_n = 0
    while True:
        actions_taken = 0
        episode_n += 1

        waterX1, waterX2, waterZ1, waterZ2 = generate_water(2)
        my_mission.drawCuboid(waterX1, 3, waterZ1, waterX2, 3, waterZ2, "water")

        # Attempt to start a mission:
        max_retries = 3
        for retry in range(max_retries):
            try:
                agent_host.startMission( my_mission, my_mission_record )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:",e)
                    exit(1)
                else:
                    time.sleep(2)

        # Loop until mission starts:
        #print("Waiting for the mission to start ", end=' ')
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            #print(".", end="")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:",error.text)

        #print()
        #print("Mission running ", end=' ')
        # ******************* START HERE ***********************

        print("Episode #" + str(episode_n))

        # Loop until mission ends:
        #fig = plt.figure()
        if episode_n % 7 == 0 or TEST_FLAG:
            print("Testing best policy")

        image = 0
        final_distance = -999
        while world_state.is_mission_running:
            #print("asdads", random.random())
            time.sleep(0.05)

            if world_state.number_of_observations_since_last_state > 0:
                obs_text = world_state.observations[-1].text
                obs = json.loads(obs_text)
                playerX = obs[u'XPos']
                playerZ = obs[u'ZPos']
                print("Old X:", playerX, "Old Z:", playerZ)
            else:
                print("failed to fetch coords")
                world_state = agent_host.getWorldState()
                continue

            if len(world_state.video_frames) > 0:
                image = process_image(world_state.video_frames[-1])

                #plt.imshow(image, cmap=plt.gray())
                #fig.canvas.draw()
                #fig.canvas.flush_events()
                #plt.show(block=False)
            else:
                print("failed to fetch image")
                world_state = agent_host.getWorldState()
                continue

            if episode_n % 7 == 0 or TEST_FLAG:
                policy_net.eval()
            else:
                policy_net.train()

            action = select_action(image, policy_net, device)
            for error in world_state.errors:
                print("Error:",error.text)
                break
            
            agent.setAction(action)
            actions_taken += 1

            time.sleep(0.5) # wait for movement to occur
        
            world_state = agent_host.peekWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                obs_text = world_state.observations[-1].text
                obs = json.loads(obs_text)
                playerX = obs[u'XPos']
                playerZ = obs[u'ZPos']
            else:
                print("failed to fetch coords")
                continue
            
            if len(world_state.video_frames) > 0:
                next_image = process_image(world_state.video_frames[-1])

                #plt.imshow(image, cmap=plt.gray())
                #fig.canvas.draw()
                #fig.canvas.flush_events()
                #plt.show(block=False)
            else:
                print("failed to fetch image")
                continue
            
            print("New X:", playerX, "New Z:", playerZ)
            final_distance = math.sqrt((playerX - (waterX1 + waterX2)/2)**2 + (playerZ - (waterZ1 + waterZ2)/2)**2)

            reward = get_reward((waterX1 + waterX2)/2, (waterZ1 + waterZ2)/2, playerX, playerZ, action)
            if episode_n % 7 == 0 and not TEST_FLAG:
                torch.save(policy_net.state_dict(), "./saved_params_final.wts")
            elif reward > 0:
                torch.save(policy_net.state_dict(), "./saved_params_train_final.wts")
            '''if reward > 0:
                if episode_n % 10 == 0:
                    torch.save(policy_net.state_dict(), './good_policy.wts')
                elif episode_n > 30:
                    torch.save(policy_net.state_dict(), './good_policy_train.wts')'''
                    
            print("Action:", agent.action_list[action] + ", Reward:", reward, "\n")
            reward = torch.tensor([reward], device=device).float()

            memory.push(image, action, next_image, reward)

            world_state = agent_host.getWorldState()

            if not world_state.is_mission_running:
                break

            if episode_n % 7 == 0 or TEST_FLAG:
                continue
            else:
                optimize_model(policy_net, target_net, optimizer, device, memory)


        #plt.close()


        print("Mission ended")
        print(f"Number of actions taken: {actions_taken}\n")
        
        # Mission has ended.

        #print("X:", str(obs[u'XPos']) + ", Z:", str(obs[u'ZPos']))

        if episode_n % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        my_mission.drawCuboid(waterX1, 3, waterZ1, waterX2, 3, waterZ2, "snow")

        with open("data_saved.txt","a") as f:
            f.write(str(final_distance) + '\n')