from autogen.agentchat.agent import Agent
from autogen.agentchat.assistant_agent import ConversableAgent
from typing import Callable, Dict, Optional, Union, List, Tuple, Any
import learn_to_pick


try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x


class LearnToPickAgent(ConversableAgent):
    """An agent that learns to pick the best response from a set of candidates."""

    ToSelectFrom = learn_to_pick.ToSelectFrom
    ToSelectFromType = learn_to_pick.base._ToSelectFrom
    BasedOn = learn_to_pick.BasedOn
    BasedOnType = learn_to_pick.base._BasedOn

    def __init__(
        self,
        name="learntopickagent",
        system_message: Optional[
            str
        ] = "You are a helpful AI assistant that remembers user teachings from prior chats.",
        human_input_mode: Optional[str] = "NEVER",
        llm_config: Optional[Union[Dict, bool]] = None,
        learn_to_pick_config: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Args:
            name (str): name of the agent.
            system_message (str): system message for the ChatCompletion inference.
            human_input_mode (str): This agent should NEVER prompt the human for input.
            llm_config (dict or False): llm inference configuration.
                Please refer to [Completion.create](/docs/reference/oai/completion#create)
                for available options.
                To disable llm-based auto reply, set to False.
            learn_to_pick_config (dict or None): Additional parameters used by LearnToPickAgent.
                To use default config, set to None. Otherwise, set to a dictionary with any of the following keys:
                - TBD
            **kwargs (dict): other kwargs in [ConversableAgent](../conversable_agent#__init__).
        """
        super().__init__(
            name=name,
            system_message=system_message,
            human_input_mode=human_input_mode,
            llm_config=llm_config,
            **kwargs,
        )
        # Register a custom reply function.
        self.register_reply(Agent, LearnToPickAgent._generate_ltp_assistant_reply, 1)

        # Assemble the parameter settings.
        self._learn_to_pick_config = {} if learn_to_pick_config is None else learn_to_pick_config
        self.verbosity = self._learn_to_pick_config.get("verbosity", 0)
        self.reset_model = self._learn_to_pick_config.get("reset_model", False)
        self.path_to_model = self._learn_to_pick_config.get("path_to_model", "./tmp/learn_to_pick_agent_model")

        # Create the rl component
        self._pick_best = learn_to_pick.PickBest.create(**self._learn_to_pick_config)

    def close(self):
        """Save the model."""
        self._pick_best.save_progress()

    def _generate_ltp_assistant_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,  # Persistent state.
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """
        """
        if messages is None:
            messages = self._oai_messages[sender]  # In case of a direct call.

        # Get the last user turn.
        def _handle_message(message):
            import json

            # If last_message is a string, attempt to parse it as JSON
            if isinstance(message, str):
                message = json.loads(message)

            # Normalize the input to the format {name: option_list/criteria_list}
            normalized_input = message.copy()

            if 'to_select_from' in message and 'based_on' in message:
                # Handle the first format
                if "name" in message['to_select_from'] and "name" in message['based_on']:
                    normalized_input[message['to_select_from']['name']] = LearnToPickAgent.ToSelectFrom(message['to_select_from']['options'])
                    normalized_input[message['based_on']['name']] = LearnToPickAgent.BasedOn(message['based_on']['options'])
                else:
                # Handle the second format    
                    normalized_input['to_select_from'] = LearnToPickAgent.ToSelectFrom(message['to_select_from'])
                    normalized_input['based_on'] = LearnToPickAgent.BasedOn(message['based_on'])
            # Handle the third format (with instantiated objects)
            else:
                for key, value in message.items():
                    if isinstance(value, LearnToPickAgent.ToSelectFromType):
                        normalized_input[key] = value
                    elif isinstance(value, LearnToPickAgent.BasedOnType):
                        normalized_input[key] = value

            return normalized_input

        last_message = messages[-1]['content']
        input = _handle_message(last_message)

        picked = self._pick_best.run(**input)

        return True, {'content': picked}

    def learn_from_user_feedback(self, score: float, response: Dict[str, Any]):
        """"""
        self._pick_best.update_with_delayed_score(score, response)
