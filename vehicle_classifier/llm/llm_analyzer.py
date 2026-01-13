import os
import asyncio
from dotenv import load_dotenv
from groq import Groq
from llm.vehicle_knowledge import VEHICLE_INFO

load_dotenv()

class LLMAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env")

        self.client = Groq(api_key=self.api_key)

        # 🔥 AI 정체성 강화
        self.system_prompt = """
너는 AI 기반 차량 이미지 분석 & 마케팅 인텔리전스 시스템이다.

너의 역할:
1. CNN, ResNet, MobileNet의 예측 결과를 해석한다.
2. 차량 유형의 실제 사용 목적과 특성을 분석한다.
3. 예측 신뢰도와 모델 간 합의도를 기반으로 판단의 신뢰성을 평가한다.
4. 해당 차량을 가장 잘 구매할 고객 유형을 추론한다.
5. 해당 차량에 가장 적합한 마케팅 전략과 광고 메시지를 생성한다.

너는 기술 분석과 비즈니스 분석을 동시에 수행하는 AI 분석가이다.
"""

    async def analyze_vehicle(self, predicted_class, confidence, model_scores):
        vehicle_info = VEHICLE_INFO.get(predicted_class, {})

        prompt = f"""
AI 차량 분류 결과:

차종: {predicted_class}
신뢰도: {confidence}

모델별 예측:
{model_scores}

차량 지식:
{vehicle_info}

다음 내용을 포함하여 분석하라:

[1] 기술 분석
- 이 차종의 일반적인 특성
- 연비, 공간, 활용 목적
- 모델 예측이 얼마나 신뢰 가능한지

[2] 고객 분석
- 이 차량을 가장 선호할 고객 유형 (나이, 가족 구성, 직업, 라이프스타일)

[3] 마케팅 전략
- 이 차량을 판매할 때 가장 효과적인 포지셔닝
- SUV/세단/트럭 중 경쟁 차량과의 차별점

[4] 광고 문구
- 이 차량에 어울리는 광고 카피 2개 생성
"""

        return await self._call_llm(prompt)

    async def analyze_model_performance(self, results):
        prompt = f"""
다음은 차량 이미지 분류 모델들의 성능이다.

{results}

모델의 안정성, 신뢰도, 실사용 적합성을 분석하라.
"""
        return await self._call_llm(prompt)

    async def generate_training_report(self, histories):
        prompt = f"""
다음은 학습 기록이다.

{histories}

과적합, 일반화 성능, 모델 신뢰성을 평가하라.
"""
        return await self._call_llm(prompt)

    async def _call_llm(self, prompt):
        loop = asyncio.get_event_loop()

        response = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
        )

        return response.choices[0].message.content
