import pandas as pd
import pandas_profiling

data = pd.read_csv('spam.csv', encoding = 'latin1')

#%%
import pandas as pd
import pandas_profiling

data = pd.read_csv('spam.csv', encoding = 'latin1')
data[:5]
# %%
pr = data.profile_report() # 프로파일링 결과 pr에 저장
# %%
pr.to_file('./pr_report.html') # pr_report.html 파일로 저장
# %%
pr