# rag_chatbot

### How to install ?

Create a .env file, provide 'GOOGLE_API_KEY=<your key>'

In Terminal or Command Prompt use :<br>`$ pip install -r requirments.txt` <br>
(before installing it is advised to use a separate env, to create or how to use [check here](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) )

we added dependencies using  <br>`$ pip freeze | awk -F'@' '{print $1}' > rag_chatbot/requirements.txt` <br>so we are confident that _there will not be any path issues_ with these @ file:///tmp/build/8754af9/toml_16161611790/work if in case you get those, please go to the requirements.txt and use this kind of regx `@.*?\/work$` to remove the unwanted stuff ! 
<br>
But still you may face issues while using howbrew _[to know more !](https://github.com/Homebrew/homebrew-core/issues/76621)_, which is not in either of the hands.
