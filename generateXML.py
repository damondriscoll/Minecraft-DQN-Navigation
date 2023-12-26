#ADD DISCRETE MOVEMENT
# https://microsoft.github.io/malmo/0.14.0/Python_Examples/Tutorial.pdf

missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                  <About>
                    <Summary>INITIALIZING</Summary>
                  </About>

                  <ModSettings>
                      <MsPerTick>150</MsPerTick>
                  </ModSettings>

                  <ServerSection>
                    <ServerInitialConditions>
                        <Time>
                            <StartTime>12000</StartTime>
                            <AllowPassageOfTime>false</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>
                    </ServerInitialConditions>
                    <ServerHandlers>
                      <FlatWorldGenerator generatorString="2;7,2x173,80;1;"/>
                      <DrawingDecorator/>
                      <ServerQuitFromTimeUp timeLimitMs="30000"/>
                      <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                  </ServerSection>

                  <AgentSection mode="Survival">
                    <Name>DropperSolver</Name>
                    <AgentStart>
                        <Placement x="0.5" y="192" z="0.5" pitch="90"/>
                    </AgentStart>
                    <AgentHandlers>
                      <ObservationFromFullStats/>
                      <ObservationFromGrid>
                          <Grid name="floor3x3">
                            <min x="-1" y="-1" z="-1"/>
                            <max x="1" y="-1" z="1"/>
                          </Grid>
                      </ObservationFromGrid>
                      <DiscreteMovementCommands/>
                      <AgentQuitFromTouchingBlockType>
                          <Block type="water"/>
                          <Block type="snow"/>
                      </AgentQuitFromTouchingBlockType>
                    </AgentHandlers>
                  </AgentSection>
                </Mission>'''

def getXML():
    return missionXML