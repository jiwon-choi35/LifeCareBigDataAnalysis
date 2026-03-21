import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------
# --- 0. 설정값: 파일 경로 및 핵심 변수명 ---
# ----------------------------------------------------
#  다운로드 받은 2020년~2024년 SAS 파일의 경로 리스트
file_paths = [
    "C:/Users/82108/Downloads/chs20_all_re/chs20_all_re.sas7bdat",
    "C:/Users/82108/Downloads/chs21_all/chs21_all.sas7bdat",
    "C:/Users/82108/Downloads/chs22_all/chs22_all.sas7bdat",
    "C:/Users/82108/Downloads/chs23_all/chs23_all.sas7bdat",
    "C:/Users/82108/Downloads/chs24_all/chs24_all.sas7bdat",
]

# 2. 확정된 변수명 및 분석 대상 연령
TARGET_AGE_MIN = 20
TARGET_AGE_MAX = 39 

VAR_AGE = 'age'                  # 연령 
VAR_DEPRESSION = 'mtb_01z1'      # 종속 변수: 현재 우울감 경험 유무
VAR_JOB_STATUS = 'soa_01z1'      # 독립 변수: 경제활동 여부 
VAR_WEIGHT = 'wt_p'              # 개인 가중치 

#  통제 변수
VAR_SEX = 'sex'                 # 통제 1: 성별 (1=남, 2=여)
VAR_DIABETES = 'dia_04z1'       # 통제 2: 당뇨병 유무 (1=예, 2=아니오)
VAR_HYPERTENSION = 'hya_04z1'   # 통제 3: 고혈압 유무 (1=예, 2=아니오)


# ---------------------------------------------
# --- 1. SAS 파일 로드 및 통합 (반복문) ---
# ---------------------------------------------
all_data = []
print("--------------------------------------------------")
print("1. 2020년부터 2024년까지 SAS 파일 로드 및 통합 시도...")


for file_path in file_paths:
    try:
        
        df_year, _ = pyreadstat.read_sas7bdat(file_path)
        
        
        df_year.columns = df_year.columns.str.lower() 
        
       
        file_name_part = os.path.basename(file_path).lower()
        year = 'N/A'
        if 'chs2' in file_name_part:
            try:
                start_index = file_name_part.find('chs') + 3
                year = int(file_name_part[start_index:start_index+2]) + 2000
            except:
                year = 'N/A'
        
        df_year['EXAMIN_YEAR'] = year
        all_data.append(df_year)
        print(f" 파일 읽기 성공: {os.path.basename(file_path)} (응답자 수: {len(df_year)})")
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 다시 확인하세요: {file_path}")
    except Exception as e:
        print(f"파일을 읽는 중 오류 발생 ({os.path.basename(file_path)}): {e}")
        
if not all_data:
    print(" 통합할 데이터가 없어 분석을 실행할 수 없습니다.")
    exit()
    
# 모든 데이터프레임을 하나로 합치기
df_combined = pd.concat(all_data, ignore_index=True)
print(f"✅ 모든 데이터 통합 완료! (총 응답자 수: {len(df_combined)})")


# ---------------------------------------------
# --- 2. 분석 대상 필터링 및 데이터 클리닝 ---
# ---------------------------------------------

# 2-1. 청년층 (20-39세) 추출
df_youth = df_combined[(df_combined[VAR_AGE] >= TARGET_AGE_MIN) & (df_combined[VAR_AGE] <= TARGET_AGE_MAX)].copy()

# 2-2. 필수 변수 결측치 제거 (통제 변수 포함)
# 우울감 (1, 2), 경제활동 (1, 2) 외 통제 변수 3가지의 유효값(1 또는 2) 확인
df_analysis = df_youth[
    (df_youth[VAR_DEPRESSION].isin([1, 2])) & 
    (df_youth[VAR_JOB_STATUS].isin([1, 2])) &
    (df_youth[VAR_SEX].isin([1, 2])) &              # 성별 결측치 제외
    (df_youth[VAR_DIABETES].isin([1, 2])) &        # 당뇨 결측치 제외
    (df_youth[VAR_HYPERTENSION].isin([1, 2]))      # 고혈압 결측치 제외
].copy()

# ---------------------------------------------
# --- 3. 변수 리코딩 및 변환 (로지스틱 회귀 준비) ---
# ---------------------------------------------

# 3-1. 종속 변수 및 독립 변수 리코딩
df_analysis['depressed'] = df_analysis[VAR_DEPRESSION].replace({2: 0, 1: 1}).astype(int)
df_analysis['econ_active'] = df_analysis[VAR_JOB_STATUS].replace({2: 0, 1: 1}).astype(int)

# 3-2.  통제 변수 리코딩: '예(1)'는 1로, '아니오/여성(2)'는 0으로 변환 
df_analysis['male'] = df_analysis[VAR_SEX].replace({2: 0, 1: 1}).astype(int) # 여성(2)을 기준(0)으로, 남성(1)을 1로
df_analysis['has_diabetes'] = df_analysis[VAR_DIABETES].replace({2: 0, 1: 1}).astype(int)
df_analysis['has_hypertension'] = df_analysis[VAR_HYPERTENSION].replace({2: 0, 1: 1}).astype(int)

# 3-3. 가중치 변수명 단순화
df_analysis = df_analysis.rename(columns={VAR_WEIGHT: 'WGT'}) 

print(f"\n2. 데이터 전처리 결과")
print(f" 최종 분석 데이터 응답자 수: {len(df_analysis)}명")
print(f"   20-39세 청년층 우울감 경험률: {df_analysis['depressed'].mean() * 100:.2f}%")

# ---------------------------------------------
# --- 4. 로지스틱 회귀 분석 실행 및 결과 출력 ---
# ---------------------------------------------

#  모델 공식 수정: 통제 변수 3가지 (성별, 당뇨, 고혈압) 모두 포함 
formula = (
    'depressed ~ econ_active + male + has_diabetes + has_hypertension'
) 

# 가중치(WGT)를 반영하여 모델 실행
model = smf.logit(formula=formula, data=df_analysis, weights=df_analysis['WGT']).fit()

print("\n--------------------------------------------------")
print("3.  다중 로지스틱 회귀 분석 최종 결과 (통제 요인 포함)")
print("--------------------------------------------------")
print(model.summary().tables[1])

# 오즈비 (Odds Ratio) 계산
odds_ratios = np.exp(model.params)
conf_int = np.exp(model.conf_int())

print("\n 오즈비(Odds Ratio) 및 95% 신뢰구간:")
odds_df = pd.DataFrame({'OR': odds_ratios, '2.5% CI': conf_int[0], '97.5% CI': conf_int[1]})
# 절편 제외 및 주요 변수만 출력
print(odds_df[['OR', '2.5% CI', '97.5% CI']].iloc[1:]) 
print("--------------------------------------------------")
