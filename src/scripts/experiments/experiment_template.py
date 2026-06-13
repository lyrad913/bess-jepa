# Import

# 제공 된 두 함수 외에는 함수를 사용하지 말것. 
def train(model, ...):
    
    return something

def do_bench(model, ...): # 혹은 do experiment
    
    return something

# 함수 안에서 작업 단위를 주석으로 잘 표현할 것. 예를 들면
"""
def do_bench(model, ...):
    # 모델에 대하여 벤치 실행
    
    # 스칼라값 정리
    
    # ~~~ 그림을 그리기.
    
    return something
"""

if __name__ == "__main__":
    # Clearml Task init, Hydra, ...
    
    # 훈련해야하는 여러 모델들에 대해서
    ... = train()
    ... = do_bench()
    
    ... = train()
    ... = do_bench()
    
    # ...
    
    # 공통으로 비교해야하거나 그림을 그려야하는 것이 있다면 여기서 위에서 얻은 값들을 활용해 하기
    
    # Report도 빼먹지 말것.