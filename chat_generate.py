from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import (ChatPromptTemplate,FewShotChatMessagePromptTemplate)
from dotenv import load_dotenv
load_dotenv()



class ChatGenerator:
    def __init__(self) -> None:
        self.chat = ChatOpenAI(temperature=0,model_name = 'gpt-4-1106-preview')
        self.example =  [
            {
            "sentence" : "쪽바리앞잡이야 너거나라가라 꺼져.",
            "masking_sentence" : "[mask]앞잡이 너거나라가라 [mask]",
            "answer":
            '''
            일본옹호자들은 일본에서 생활하는게 나을 것 같네요
            ''',
            "sentence" : "미친 씨발 개슬람년아 그럼 한국에 오질 말든가",
            "masking_sentence" : "미친 [mask] [mask]슬람년아 그럼 한국에 오질 말든가",
            "answer":
            '''
            이슬람 여성분은 그러시면 한국에 오지않는게 좋을듯해요.
            '''
            },           
            {"sentence" : "음주운전하는 새끼들은 진짜 대가리에 뭐가 든건지... 다 무기징역 시켜라",
            "masking_sentence" : "음주운전하는 [mask] 진짜 [mask] 뭐가 든건지... 다 무기징역 시켜라",
            "answer":
            '''
            음주운전하는 사람들은 진짜 머리에 뭐가 든건지... 다 무기징역 시켜라
            '''
                        },
            {
            "sentence" : "좌빨 영화 납시요 개돼지들 선동시키기 딱이요",
            "masking_sentence" : "[mask] 영화 납시요 [mask] [mask]시키기 딱이요",
            "answer":
            '''
            좌파 영화 납시요. 사람들 부추기기 딱이요 
            '''
            }
           
        ]
        self.example_prompt = ChatPromptTemplate.from_messages(
                        [
                ("human", "원문 :{sentence}\n 마스킹 문장 : {masking_sentence}"),
                ("ai", "순화 문장 : {answer}"),
            ]
        )
        self.few_shot_prompt =  FewShotChatMessagePromptTemplate(
            example_prompt=self.example_prompt,
            examples=self.example,
        )
        self.final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "너는 문장 스타일 변환을 하는 역할을 할거야.  해당 댓글이 악성 댓글로 판단되는데 중요한 영향을 미친(=feature importance가 높은) 단어를 [mask] 처리한 문장이 마스킹 문장으로 주어지면 두 문장을 바탕으로 기존 댓글의 문맥은 유지하면서 순화 댓글을 생성하는 것이 너의 임무야."),
                self.few_shot_prompt,
                ("human", "원문 :{sentence}\n 마스킹 문장 : {masking_sentence}"),
            ]
        )
        
        self.chain = self.final_prompt | self.chat
    def covert_sentence(self,ori_setence,mask_sentence):
        return self.chain.invoke({"sentence" : {ori_setence},
                           "masking_sentence" : {mask_sentence}})
