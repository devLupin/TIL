<!--
stories.md 에서 정의한 정책들을 대응하기 위한 문구들을 정의함.

대쉬(-) 이후 나오는 문구들은 전부 intent 항목으로 처리하는 뜻인 거 같음. 아직은 확실히 모름.
-->

## intent:greet
- Hi
- Hey
- Hi bot
- Hey bot
- Hello
- Good morning
- hi again
- hi folks

## intent:bye
- goodbye
- goodnight
- good bye
- good night
- see ya
- toodle-oo
- bye bye
- gotta go
- farewell

## intent:thank
- Thanks
- Thank you
- Thank you so much
- Thanks bot
- Thanks for that
- cheers


<!-- 접두사/ 를 가지면 ResponseSelector에 의해 접두사 의도로 인식됨. -->

## intent: faq/ask_channels
- What channels of communication does rasa support?
- what channels do you support?
- what chat channels does rasa uses
- channels supported by Rasa
- which messaging channels does rasa support?

## intent: faq/ask_languages
- what language does rasa support?
- which language do you support?
- which languages supports rasa
- can I use rasa also for another laguage?
- languages supported

## intent: faq/ask_rasax
- I want information about rasa x
- i want to learn more about Rasa X
- what is rasa x?
- Can you tell me about rasa x?
- Tell me about rasa x
- tell me what is rasa x