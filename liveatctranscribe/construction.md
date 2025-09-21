# This file is to be used in the reconstruction of the informed fast atc transcribe engine (ifa_transcriber.py, ifa_components.py) if needed


look at liveatctranscribe, your job is to take fastatc_transcriber.py and work on ifa_transcriber.py. Our end goal is to combine the live ATC transcription component and pair it with our api fetched knowledge of the plane positions. Let us combine this data for the goal of understanding for a state S1 for an airport, what will the ATC ask what plane to do? Please understand that our end goal is to generate a validation data set for an agent to do  


test: planes are in X state; validate that for X state, the AI agent responds with Y action in correspondence with the same Y action that a real life ATC agent took. Know that our AI agent is to be a bare LLM but its system prompt and corresponding scaffolding will evolve, but that is not too much of your concern. To reiterate your goal is to at first construct a set of test cases that align airport state with the action an ATC operator took. I will be running ifa_transcriber and playing in my blackhole 2h driver microphone the live audio of the ATC tower. Let me know if you have any questions. As this is a complex task, ask me as many questions as you need to ground your thinking, proceed incrementally, and carefully. Reference fastatc_transcriber.py and client.py OFTEN.  Please, thank you, and goodluck


>Correlate commands with aircraft state -> consider  that the live transcription will already be timed well with the input of api data, but note that since cereberus and whisper and TTL add some lag, ensure you correspond the timing of the input of flight data and the input of transcription closest with the concept that the flight data packet A and the speech segment B will arrive at the same time T, but the interpretation of speech segment B will mean we have an interpretation of the commands ATC issues at the airport state described in packet A at time B + K where K is the processing time of speech segment B, suggest some remediation methods and lets discuss further. 

>Should ifa_transcriber.py run both systems simultaneously (ATC transcription + aircraft monitoring)? -> Yes

> How frequently should I capture aircraft state snapshots?
-> Whenever the speech segment B is finished processing we can look at the airport data packet A's successor and B's successor. If you are confused with this instruction let me know


> Are you focusing on specifically JFK airport operation?
- > Yes, and every system prompt to the ifa_transcriber should be written to that end, however there may be more features you can suggest to improve performance on JFK operation 

>Validation data set type:
-> JSON

> Each record: 
-> Propose what you think is best and justify, however I'm inclined towards {timestamp, aircraft_states, atc_command, command_type, affected_aircraft}  Remember we are designing an AI agent to recieve aircraft_states exactly as was seen in client.py and others, and to make the suggestions as a human ATC operator would, and to identify chokepoints before they happen among others but for now this is what we will focus on

> How should I handle unclear/partial transcriptions? 
-> Please try to stick to what was written (and works) in fastatc_transcriber already for the transcription part, but know that we will work on improving transcription clarity later. 

> Should I parse and categorize ATC commands (e.g., "taxi", "cleared to land", "contact departure")?
-> Yes, design this part of the system how you in your expert opinion believe is most reliable, and viably performant


>Should I include airport-wide state (all aircraft) or focus on specific aircraft mentioned in commands?
-> All JFK planes landing and near the ground

Please note any further questions you have before we proceed

