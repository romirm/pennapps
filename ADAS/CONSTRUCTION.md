Constructing this based on Automated Design of Agentic Systems Research Paper


https://arxiv.org/pdf/2408.08435 , although I've also placed it in the wd as ADAS.pdf


The general gist is have one agent, the all powerful, highly inteligent meta agent, and another the task agent. 


The operation is the meta agent will take the task agent, propose N variants/mutations of the task agent, test eaches performance with the validation datasets in ./validation-dataset, grab the best one, and repeat. 

The task agent's job should be able to respond well to situations. The overarching goal of the super-project that encapsulates this project is to create an AI agent that can identify bottlenecks before they happen and respond well. We define a bottleneck and identify future bottlenecks with a seperate model. The job of the task agent is to read the situation, understand that there is a bottleneck, and perform movements as an air traffic controller would based on what is tested in the validation-dataset. To assist in this task, I recommend the task agent also summarize what it sees in the validation set, interpret the latitudes and longitudes before it makes its assessment. 


To be clear, the validation set consists of a json you must study, but the goal is, given the plane positions, you should perform the same action as was interpreted from the pilot transcripts in the 


As an aside, we will have an evaluator agent assisting the meta agent, This evaluator performs the LLM-as-a-judge role to ensure the task agent IS doing a good job. For a given airport state, the evaluator agent tests that the task agent has proposed the correct action. 

As a furtherance I will also suggest that there be a cleaning agent that removes validation cases from raw-validation-dataset for any reason like
a low confidence of assessment, a low chance of actually being an instruction etc, a vague command. it has to be a fair and coherent adversarial agent, but you may propose a better system as is appropriate When the dataset is cleaned, the files shall be moved into the validation-dataset folder. 

