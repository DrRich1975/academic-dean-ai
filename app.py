import io
import re
from difflib import SequenceMatcher
from docx import Document
from pathlib import Path
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title='Academic Dean AI',page_icon='🎓',layout='wide')
st.title('🎓 Academic Dean AI')
st.caption('Academic Leadership Command Center · class prototype · no external AI API required')
st.caption('Demo information is fictional unless you upload your own file. Uploaded documents are processed only in this app session.')

REQUIRED=['Academic Org Campus','Class Status','Loc Short Desc','Class Nbr','Tot Enrl','Cap Enrl']
DAYS=['M','T','W','R','F','S','U']
DAY_NAMES=dict(zip(DAYS,['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']))

@st.cache_data(show_spinner=False)
def read_data(data:bytes, filename:str):
    f=io.BytesIO(data)
    if filename.lower().endswith('.csv'):
        return pd.read_csv(f,dtype={'Class Nbr':str})
    return pd.read_excel(f,sheet_name='Schedule',engine='openpyxl',dtype={'Class Nbr':str})

def norm_days(raw):
    if pd.isna(raw): return set()
    x=str(raw).upper().strip().replace('TH','R').replace('TU','T').replace('SA','S').replace('SU','U')
    if x in ('NAN','NONE','ONLINE','-',''):return set()
    return {d for d in DAYS if d in x}

def hours_to_minutes(v):
    if pd.isna(v):return None
    if isinstance(v,(datetime,pd.Timestamp)):return v.hour*60+v.minute
    if hasattr(v,'hour') and hasattr(v,'minute'):return v.hour*60+v.minute
    if isinstance(v,(int,float,np.number)):
        if 0<=float(v)<1: return round(float(v)*1440)
        return None
    s=str(v).strip()
    try:
        t=pd.to_datetime(s).time();return t.hour*60+t.minute
    except Exception:return None

def clean(df):
    missing=[c for c in REQUIRED if c not in df]
    if missing: raise ValueError('Missing CMR columns: '+', '.join(missing))
    d=df.copy()
    for c in ['Tot Enrl','Cap Enrl','Max Units']:
        if c in d:d[c]=pd.to_numeric(d[c],errors='coerce')
    base=d[(d['Academic Org Campus'].astype(str).str.strip()=='HOME') & (d['Class Status'].astype(str).str.strip()=='A')].copy()
    # Exclude all automatic lab components marked unlimited 999/9999, regardless of future department labels.
    excluded=base[base['Cap Enrl'].isin([999,9999])].copy()
    d=base[~base['Cap Enrl'].isin([999,9999])].copy()
    d['Location Category']=np.where(d['Loc Short Desc'].fillna('').astype(str).str.strip()=='Homestead','On campus','Off campus / other')
    d['Instructional Modality']=d.get('Instr Mode',pd.Series('Unknown',index=d.index)).fillna('Unknown').astype(str)
    d['Scheduled Days']=d.get('Concat Days',pd.Series('',index=d.index)).map(norm_days)
    d['Start Minutes']=d.get('Mtg Start',pd.Series(index=d.index,dtype=object)).map(hours_to_minutes)
    d['End Minutes']=d.get('Mtg End',pd.Series(index=d.index,dtype=object)).map(hours_to_minutes)
    d['Credits (approx)']=d['Tot Enrl']*d.get('Max Units',pd.Series(np.nan,index=d.index))
    d['Fill Rate']=np.where(d['Cap Enrl']>0,d['Tot Enrl']/d['Cap Enrl'],np.nan)
    # The schedule worksheet may contain multiple instructor rows for same section.
    unique_keys=['Term','Class Nbr'] if 'Term' in d else ['Class Nbr']
    dupe_count=int(d.duplicated(unique_keys).sum())
    d=d.drop_duplicates(unique_keys,keep='first')
    return d,excluded,len(base),dupe_count

def is_physical(row):
    mode=str(row['Instructional Modality']).upper()
    # Online, LIVE remote, and unscheduled asynchronous sections should never be counted as physical presence.
    return not ('ONLINE' in mode or mode.startswith('ON ') or 'MDC LIVE' in mode or mode.startswith('LIVE') or 'REMOTE' in mode)

def time_label(minutes):
    h,m=divmod(int(minutes),60);return f'{(h-1)%12+1}:{m:02} {"AM" if h<12 else "PM"}'

def show_metrics(d):
    a,b,c,e=st.columns(4)
    a.metric('Qualifying sections',f'{len(d):,}')
    b.metric('Section enrollments',f'{d["Tot Enrl"].sum():,.0f}')
    c.metric('Estimated credit hours',f'{d["Credits (approx)"].sum():,.0f}',help='Enrollment × Max Units; approximate, not certified credit-hour reporting')
    valid=d[d['Cap Enrl'].between(1,998)]
    e.metric('Weighted fill rate',f'{valid["Tot Enrl"].sum()/valid["Cap Enrl"].sum():.1%}' if len(valid) and valid['Cap Enrl'].sum() else 'N/A')

st.sidebar.header('Data')
files=st.sidebar.file_uploader('Upload fall and/or spring CMR (Schedule sheet)',type=['xlsx','xlsm','csv'],accept_multiple_files=True)
st.sidebar.info('Includes only Class Status A + Academic Org Campus HOME. Excludes unlimited-capacity auto lab sections (999/9999). "Homestead" ≠ "HOMESTEAD".')

if not files:
    if st.sidebar.button('Load fictional demonstration data',type='primary'):
        st.session_state['use_demo']=True
    if st.session_state.get('use_demo'):
        demo=Path(__file__).parent/'sample_cmr.csv'
        class DemoFile:
            name='sample_cmr.csv'
            def getvalue(self): return demo.read_bytes()
        files=[DemoFile()]
    else:
        st.info('Upload a CMR Excel file, or click **Load fictional demonstration data** in the sidebar. No institutional source data are bundled.')
        st.stop()

loaded={}
for file in files:
    try:
        with st.spinner(f'Reading {file.name}...'):
            raw=read_data(file.getvalue(),file.name)
            d,excluded,base_n,dupes=clean(raw)
        loaded[file.name]=(d,excluded,base_n,dupes)
    except Exception as exc: st.error(f'{file.name}: {exc}')
if not loaded:st.stop()
selected=st.sidebar.selectbox('Analyze report',list(loaded))
d,excluded,base_n,dupes=loaded[selected]
st.success(f'Filters verified: HOME + A. {base_n} eligible HOME/A rows; {len(excluded)} unlimited-capacity lab rows removed; {dupes} repeated section rows removed; {len(d)} analyzed sections.')
show_metrics(d)

pages=st.tabs(['📈 Enrollment intelligence','🏫 Location & schedule','🧑‍🏫 Faculty & sections','📆 Projects & grants','💼 Grant portfolio','📝 Monthly report','🔁 Initiative progress','🎯 Strategic goals','🔎 Report review'])
with pages[0]:
    st.subheader('Enrollment and academic performance')
    group_choices=[c for c in ['Acad Org Name','Course Prefix (Subject)','Instructional Modality','Location Category','Loc Short Desc'] if c in d]
    group=st.selectbox('Group results by',group_choices)
    summary=d.groupby(group,dropna=False).agg(Sections=('Class Nbr','size'),Enrollments=('Tot Enrl','sum'),Capacity=('Cap Enrl','sum'),Approx_Credit_Hours=('Credits (approx)','sum')).reset_index()
    summary['Fill Rate']=np.where(summary.Capacity>0,summary.Enrollments/summary.Capacity,np.nan)
    st.dataframe(summary.sort_values('Enrollments',ascending=False),use_container_width=True,hide_index=True)
    st.bar_chart(summary.set_index(group)['Enrollments'])
    if len(loaded)>1:
        other=st.selectbox('Compare with another report',[x for x in loaded if x!=selected])
        st.warning('Cross-term comparisons such as Fall 2026 vs Spring 2027 are NOT year-over-year comparisons. Compare like terms and equivalent registration snapshots before drawing trend conclusions.')
        old=loaded[other][0]
        if group in old:
            x=d.groupby(group)['Tot Enrl'].sum().rename('Current');y=old.groupby(group)['Tot Enrl'].sum().rename('Comparison')
            comp=pd.concat([x,y],axis=1).fillna(0)
            comp['Difference']=comp.Current-comp.Comparison
            comp['Percent Change']=np.where(comp.Comparison!=0,comp.Difference/comp.Comparison,np.nan)
            st.caption('Raw section enrollment totals, not distinct students; term season and report timing may differ.')
            st.dataframe(comp.sort_values('Difference'),use_container_width=True)
    st.subheader('Sections needing review')
    low=st.slider('Show fill rates below',10,100,50,step=5)
    weak=d[(d['Cap Enrl'].between(1,998))&(d['Fill Rate']<low/100)]
    st.dataframe(weak[[x for x in ['Class Nbr','Course Prefix (Subject)','Course Number (Catalog Nbr)','Class Descr','Acad Org Name','Loc Short Desc','Tot Enrl','Cap Enrl','Fill Rate'] if x in d]].sort_values('Fill Rate'),use_container_width=True,hide_index=True)
    st.subheader('Machine-learning anomaly flags')
    eligible=d[d['Cap Enrl'].between(1,998)&d['Tot Enrl'].notna()].copy()
    if len(eligible)>=15:
        X=eligible[['Tot Enrl','Cap Enrl','Fill Rate']].replace([np.inf,-np.inf],np.nan).dropna()
        if len(X)>=15:
            model=IsolationForest(n_estimators=100,contamination=0.06,random_state=42)
            flags=model.fit_predict(X)
            out=eligible.loc[X.index].loc[flags==-1]
            st.dataframe(out[[x for x in ['Class Nbr','Class Descr','Tot Enrl','Cap Enrl','Fill Rate','Acad Org Name'] if x in out]],use_container_width=True,hide_index=True)
            st.caption('Unsupervised Isolation Forest flags numerical outliers for review; flags are not proof of an error or a negative outcome.')
    else:st.info('At least 15 sections with usable enrollment/capacity fields are required for anomaly detection.')
with pages[1]:
    st.subheader('Location versus modality')
    st.dataframe(d.groupby(['Location Category','Instructional Modality'],dropna=False).agg(Sections=('Class Nbr','size'),Enrollments=('Tot Enrl','sum')).reset_index(),hide_index=True,use_container_width=True)
    st.caption('Only exact case-sensitive Loc Short Desc = "Homestead" is classified on-campus. All other labels, including "HOMESTEAD", are off-campus/other.')
    st.subheader('Estimated scheduled campus activity')
    day=st.selectbox('Day of week',list(DAY_NAMES),format_func=lambda x:DAY_NAMES[x])
    minute=st.slider('Time of day',min_value=420,max_value=1320,value=660,step=15,format='%d')
    st.caption('Selected time: '+time_label(minute))
    active=d[(d['Location Category']=='On campus')&d.apply(is_physical,axis=1)&d['Scheduled Days'].map(lambda x:day in x)&(d['Start Minutes']<=minute)&(d['End Minutes']>minute)]
    st.metric('Scheduled section enrollments at this time',f'{active["Tot Enrl"].sum():,.0f}')
    st.metric('Scheduled in-person sections',len(active))
    columns=[x for x in ['Class Nbr','Class Descr','Acad Org Name','Loc Short Desc','Facility ID','Concat Days','Mtg Start','Mtg End','Instr Mode','Tot Enrl'] if x in active]
    st.dataframe(active[columns],hide_index=True,use_container_width=True)
    st.warning('Not actual occupancy or a real-time student locator. Enrollment is section-based, not deduplicated people; attendance, cancellations, holidays, and changes to meeting arrangements are not reflected.')
with pages[2]:
    st.subheader('Faculty assignments — initial Schedule worksheet view')
    col='Instructor' if 'Instructor' in d else ('Instr Name' if 'Instr Name' in d else ('Unnamed: 55' if 'Unnamed: 55' in d else None))
    # CMR schedule has additional instructor columns; prefer BD (Excel column 56) when present.
    if col is None and len(d.columns)>55:col=d.columns[55]
    if col:
        d2=d.copy();d2['Faculty']=d2[col].fillna('Unassigned').astype(str).str.strip().replace({'':'Unassigned'})
        fac=d2.groupby('Faculty').agg(Sections=('Class Nbr','size'),Section_Enrollments=('Tot Enrl','sum')).reset_index().sort_values('Sections',ascending=False)
        st.dataframe(fac,hide_index=True,use_container_width=True)
        st.caption('Preliminary section assignment view only; team teaching and the dedicated Instructors worksheet are not yet reconciled. Do not use as official faculty workload or payroll calculations.')
    else:st.info('Instructor assignment data not found on Schedule sheet.')
    st.download_button('Download filtered section records (CSV)',data=d.drop(columns=['Scheduled Days']).to_csv(index=False),file_name='homestead_active_filtered.csv')
with pages[3]:
    st.subheader('Strategic initiatives and grant tracker')
    st.caption('Class prototype: editable records held in this browser session only. Download updates to retain them.')
    if 'projects' not in st.session_state:st.session_state.projects=pd.DataFrame([
        {'Project':'Faculty innovation initiative','Category':'Strategic initiative','Owner':'Academic Affairs','Deadline':'2026-11-15','Status':'In progress','Notes':'Illustrative sample'},
        {'Project':'Grant quarterly narrative','Category':'Grant','Owner':'Project director','Deadline':'2026-11-30','Status':'Needs review','Notes':'Illustrative sample'},
        {'Project':'Dual enrollment partnership review','Category':'Enrollment','Owner':'Academic Affairs','Deadline':'2026-12-10','Status':'Planned','Notes':'Illustrative sample'}])
    edited=st.data_editor(st.session_state.projects,num_rows='dynamic',use_container_width=True,key='project_editor')
    st.session_state.projects=edited
    due=edited.copy();due['Deadline parsed']=pd.to_datetime(due['Deadline'],errors='coerce')
    days=(due['Deadline parsed']-pd.Timestamp.today().normalize()).dt.days
    due['Priority']=np.select([days<0,days<=14],['Overdue','Due soon'],default='Upcoming')
    st.dataframe(due[['Project','Owner','Deadline','Status','Priority','Notes']],hide_index=True,use_container_width=True)
    st.download_button('Save project tracker (CSV)',edited.to_csv(index=False),file_name='academic_projects.csv',mime='text/csv')
with pages[4]:
    st.subheader('Campus grant portfolio: oversight and financial snapshot')
    st.caption('Upload a local campus grants spreadsheet. No real financial records are bundled with the public demo. These figures are a snapshot, not grant compliance or performance determinations.')
    grant_file=st.file_uploader('Grant/projects budget report (.xlsx)',type=['xlsx'],key='grants_upload')
    if grant_file:
        try:
            grant_raw=pd.read_excel(io.BytesIO(grant_file.getvalue()),header=0,engine='openpyxl')
            needed=['Department','Fund','Project','Sum of Budget','Sum of Expense','Sum of Encumbrance','Sum of Pre-Encumbrance','Sum of Available Budget*']
            missing=[x for x in needed if x not in grant_raw.columns]
            if missing: raise ValueError('Missing expected budget columns: '+', '.join(missing))
            ledger=grant_raw.loc[grant_raw['Project'].notna()].copy()
            ledger['Department']=grant_raw['Department'].ffill().loc[ledger.index]
            ledger['Fund']=grant_raw['Fund'].ffill().loc[ledger.index]
            for field in needed[3:]:ledger[field]=pd.to_numeric(ledger[field],errors='coerce').fillna(0)
            ledger=ledger.rename(columns={'Sum of Budget':'Budget','Sum of Expense':'Expense','Sum of Encumbrance':'Encumbrance','Sum of Pre-Encumbrance':'Pre-Encumbrance','Sum of Available Budget*':'Available Budget'})
            ledger['Budget Utilization (expensed)']=np.where(ledger['Budget']>0,ledger['Expense']/ledger['Budget'],np.nan)
            ledger['Reconciliation difference']=ledger['Budget']-ledger[['Expense','Encumbrance','Pre-Encumbrance','Available Budget']].sum(axis=1)
            ledger['Oversight']='To be confirmed'
            ledger['Principal Investigator / Director']='To be confirmed'
            ledger['Award / Performance Dates']='To be confirmed'
            ledger['Reporting Deadlines']='To be confirmed'
            ledger['Next Action']='Confirm oversight and reporting requirements'
            a,b,c=st.columns(3)
            a.metric('Budget lines / projects',len(ledger))
            b.metric('Total recorded budget',f"${ledger['Budget'].sum():,.0f}")
            c.metric('Total available balance',f"${ledger['Available Budget'].sum():,.0f}")
            st.warning('Project rows may include funding splits, matching funds, or allocations. Do not interpret line count as number of distinct grants; do not sum as unique grant awards without reconciliation.')
            st.dataframe(ledger,use_container_width=True,hide_index=True)
            st.download_button('Download grant follow-up template',ledger.to_csv(index=False).encode('utf-8-sig'),file_name='grant_portfolio_follow_up.csv',mime='text/csv')
            if (ledger['Reconciliation difference'].abs()>1).any():st.warning('Some budget lines do not reconcile exactly. Check rounding, extract timing, or source definitions.')
        except Exception as exc:st.error(f'Unable to read grant report: {exc}')
    else:
        st.info('Your October 9 grant report can be uploaded here privately during your session. The public classroom demo uses no actual grant finances.')
    st.markdown('**Oversight questions to complete later:** Academic Affairs responsibility; director/PI; award and end dates; required financial and narrative reports; deliverables; match requirements; restricted spending; upcoming renewals; corrective actions.')
with pages[5]:
    st.subheader('Monthly Academic Affairs Update Builder')
    st.caption('Structured draft tool; full Word-document comparison and personalized generative rewriting will be added after reviewing original/final report pairs.')
    month=st.text_input('Reporting month',value=date.today().strftime('%B %Y'))
    accomplishments=st.text_area('Major accomplishments (bullet notes)',height=110)
    challenges=st.text_area('Challenges / pending decisions',height=85)
    priorities=st.text_area('Priorities for next month',height=85)
    use_summary=st.checkbox('Include verified CMR snapshot metrics',value=True)
    lines=[f'ACADEMIC AFFAIRS MONTHLY UPDATE — {month}','', 'KEY ACCOMPLISHMENTS',accomplishments or '(Complete before submission)','', 'CHALLENGES / DECISIONS NEEDED',challenges or '(Complete before submission)','', 'UPCOMING PRIORITIES',priorities or '(Complete before submission)']
    if use_summary:lines+=['','ACADEMIC DATA SNAPSHOT',f'Eligible active HOME sections (excluding automatic unlimited labs): {len(d):,}',f'Section enrollments (not unduplicated students): {d["Tot Enrl"].sum():,.0f}',f'Estimated credit hours (enrollment × Max Units): {d["Credits (approx)"].sum():,.0f}']
    report='\n'.join(lines)
    st.text_area('Editable draft preview (copy to ChatGPT for personalized editing)',value=report,height=320)
    st.download_button('Download report draft (.txt)',report,file_name='academic_affairs_monthly_draft.txt',mime='text/plain')
    st.info('Before submission, check quantitative claims, missing achievements, grant commitments, and confidentiality. This template does not invent accomplishments.')


with pages[6]:
    st.subheader('Strategic Initiative Progress Tracker')
    st.caption('Track changes across reporting months instead of repeating that work continued. All seeded records below are fictional demonstration examples. Your edits are held in the browser session; download the CSV to keep them.')
    if 'progress_log' not in st.session_state:
        st.session_state.progress_log = pd.DataFrame([
            {'Initiative':'Civic literacy support','Reporting month':'2026-07','Previous milestone':'Planning instructional resources','New evidence / milestone':'Held a student preparation workshop','Status':'Progress','Next action':'Measure participation and exam outcomes','Owner':'Academic Affairs'},
            {'Initiative':'Civic literacy support','Reporting month':'2026-08','Previous milestone':'Student preparation workshop','New evidence / milestone':'Completed a shared quiz bank and faculty resources','Status':'New milestone','Next action':'Collect faculty adoption feedback','Owner':'Academic Affairs'},
            {'Initiative':'Civic literacy support','Reporting month':'2026-09','Previous milestone':'Shared materials completed','New evidence / milestone':'','Status':'Needs update','Next action':'Ask for usage and student outcome data','Owner':'Academic Affairs'},
            {'Initiative':'Workforce pathway pilot','Reporting month':'2026-07','Previous milestone':'Program concept approved','New evidence / milestone':'Drafted partner responsibilities and eligibility criteria','Status':'Progress','Next action':'Confirm partner agreement','Owner':'Program leadership'},
            {'Initiative':'Workforce pathway pilot','Reporting month':'2026-08','Previous milestone':'Partner responsibilities drafted','New evidence / milestone':'','Status':'Ongoing','Next action':'Ask whether agreement was completed','Owner':'Program leadership'},
        ])
    log=st.file_uploader('Optional: load your previously saved progress tracker CSV',type=['csv'],key='progress_upload')
    if log is not None and st.session_state.get('progress_loaded_file') != (log.name,len(log.getvalue())):
        try:
            new=pd.read_csv(log)
            if set(st.session_state.progress_log.columns).issubset(new.columns):
                st.session_state.progress_log=new[st.session_state.progress_log.columns].copy()
                st.session_state['progress_loaded_file']=(log.name,len(log.getvalue()))
            else: st.error('Uploaded tracker is missing required columns.')
        except Exception as exc:st.error(f'Unable to load tracker: {exc}')
    progress=st.data_editor(st.session_state.progress_log,num_rows='dynamic',use_container_width=True,key='progress_editor')
    st.session_state.progress_log=progress
    st.download_button('Save initiative history (CSV)',progress.to_csv(index=False),file_name='initiative_progress_history.csv',mime='text/csv')
    valid=progress.copy()
    valid['Reporting month']=valid['Reporting month'].fillna('').astype(str)
    names=sorted(valid['Initiative'].dropna().astype(str).unique())
    if names:
        selected_initiative=st.selectbox('Review history for an initiative',names)
        history=valid[valid['Initiative'].astype(str)==selected_initiative].sort_values('Reporting month')
        st.dataframe(history[['Reporting month','Previous milestone','New evidence / milestone','Status','Next action']],hide_index=True,use_container_width=True)
        latest=history.iloc[-1]
        evidence=str(latest.get('New evidence / milestone','')).strip()
        if evidence and evidence.lower()!='nan':
            st.success('Potential reportable change: '+evidence)
            st.write('**Suggested concise update:** '+selected_initiative+': '+evidence.rstrip('.')+'.')
        else:
            st.warning('No new milestone documented for the latest month. Do not repeat a generic ongoing statement as an accomplishment.')
            st.write('**Question for the initiative owner:** What specific milestone was completed, decision was made, measurable result emerged, or barrier changed since the previous report?')
        st.write('**Recorded next action:** '+str(latest.get('Next action','')))
    st.markdown('**Reporting rule:** Report a milestone, a meaningful change in risk, a new decision, or a measurable outcome. Otherwise, carry the initiative forward internally and ask for an update. A missing update is not evidence of project failure.')


with pages[7]:
    st.subheader('Historical strategic goals: 2025–2026')
    st.caption('Source: Summer 2026 Homestead Campus Strategic Goals Report Out. The 2026–2027 goals have not yet been adopted. Do not infer complete target attainment from an accomplishment narrative.')
    historical=pd.DataFrame([
        {'Goal':'Strengthen the Dual Enrollment Experience','Target / commitment':'Campus task force; mentorship, academic support and parent engagement for post-DE continuation','Documented evidence':'College-going intention 77.8% to 83.3% in survey responses; 664 degree-seeking registrations cited for fall cohort','Verification / next question':'Confirm comparable survey samples and matriculation denominator'},
        {'Goal':'Enhance Teaching and Learning','Target / commitment':'Faculty innovation program in at least 10 classes with at least 10 faculty','Documented evidence':'Teaching & Innovation Toolkit launched; 6+ lesson examples; innovative practices documented from 8+ faculty','Verification / next question':'Verify final faculty and class participation counts'},
        {'Goal':'Student Empowerment Through Community Engagement','Target / commitment':'Student-led community initiatives with reflection on empathy each major term','Documented evidence':'America 250 oral histories / local history projects; 87.5% survey agreement on empathy understanding','Verification / next question':'Confirm participation volume and survey sample'},
        {'Goal':'Expand Career-Connected Learning & Credential Completion','Target / commitment':'Micro-internships and embedded career/college credentials','Documented evidence':'Consulting-based course experience substituted for micro-internships; 99 CCCs added, 12 pending','Verification / next question':'Clarify what 99 CCCs represents and confirm subsequent credential awards'},
        {'Goal':'Advance Campus Sustainability & Operational Efficiency','Target / commitment':'Budget training each semester and proposals funding strategic initiatives','Documented evidence':'7 grant applications, 1 awarded and 1 pending as reported; budget training offerings','Verification / next question':'Reconcile applications and awards to project ledger; confirm training attendance'},
    ])
    st.dataframe(historical,use_container_width=True,hide_index=True)
    st.download_button('Download historical goal archive',historical.to_csv(index=False).encode('utf-8-sig'),file_name='historical_goals_2025_2026.csv',mime='text/csv')
    st.subheader('2026–2027 goal planning workspace')
    st.info('No approved 2026–2027 campus goals supplied yet. These rows are blank planning fields, not adopted targets.')
    if 'new_goals' not in st.session_state:
        st.session_state.new_goals=pd.DataFrame(columns=['Proposed Goal','Measure / baseline','Target','Owner','Due date','Milestone','Evidence source','Approval status'])
    draft_goals=st.data_editor(st.session_state.new_goals,num_rows='dynamic',use_container_width=True,key='new_goals_editor')
    st.session_state.new_goals=draft_goals
    st.download_button('Export proposed goals',draft_goals.to_csv(index=False),file_name='proposed_2026_2027_goals.csv',mime='text/csv')


def docx_items(data):
    document=Document(io.BytesIO(data))
    chunks=[]; section='Uncategorized'
    headings=['Enrollment and Productivity Update','Status of Program Development','Status of New Grant Development','Update on Industry and Community Partnerships','Engagement with Feeder Schools','Professional Development Activities','Priorities for Next Month','Academic Affairs','Student Services','Campus Administrative Services','Other Campus Events']
    for para in document.paragraphs:
        t=' '.join(para.text.split())
        if not t:continue
        if any(t.casefold().startswith(h.casefold()) for h in headings) and len(t)<100:
            section=t; continue
        if t.startswith(('MDC Homestead','Academic Affairs Updates')) or re.fullmatch(r'(May|June|July|August|September|October|November|December|January|February|March|April) 20\d{2}',t):continue
        if len(t)>25: chunks.append({'Source heading':section,'Item':t,'Category':'','Suggested action':'','Reason':''})
    for table in document.tables:
        for row in table.rows[1:]:
            parts=[' '.join(c.text.split()) for c in row.cells]
            t=' | '.join(p for p in parts if p)
            if len(t)>35:chunks.append({'Source heading':'Table / professional development','Item':t,'Category':'','Suggested action':'','Reason':''})
    return chunks

KEYWORDS={
    'Results / metrics':['increas','decreas','percent','%','enrollment','retention','completion','pass rate','fill rate','credit hour','graduates'],
    'Grant / investment':['grant','awarded','funding','nsf','title iii','budget','proposal','perkins'],
    'Program / workforce':['launch','program development','credential','certificate','aviation','faa','nursing','workforce','dual enrollment','partnership'],
    'Student / faculty recognition':['recognized','award','research','published','scholarship','mentor of the year'],
}
def classify_item(t):
    low=t.lower();hits=[name for name,words in KEYWORDS.items() if any(word in low for word in words)]
    generic=['attended the','participated in the meeting','continued to monitor','continued coordination','regular meeting']
    if hits:return hits[0],'Review / potential highlight','Mentions '+', '.join(hits)+'; confirm outcome, significance and reporting month.'
    if any(x in low for x in generic):return 'Routine activity','Condense / request outcome','Activity alone may not demonstrate a new milestone; ask what changed.'
    return 'Other','Review / verify','Assess relevance, timeliness, and relationship to campus objectives.'

with pages[8]:
    st.subheader('Word reportability review')
    st.caption('Rule-assisted triage based on your editorial examples. This is not an embedded ChatGPT model: no semantic rewriting or automatic deletion; final editorial decisions remain yours.')
    draft_doc=st.file_uploader('Upload assistant monthly draft (.docx)',type=['docx'],key='draft_doc')
    prior_doc=st.file_uploader('Optional: upload prior-month approved update (.docx) to flag repeated descriptions',type=['docx'],key='prior_doc')
    if draft_doc:
        try:
            items=docx_items(draft_doc.getvalue())
            past=[x['Item'] for x in docx_items(prior_doc.getvalue())] if prior_doc else []
            for item in items:
                cat,action,reason=classify_item(item['Item'])
                item.update({'Category':cat,'Suggested action':action,'Reason':reason})
                if past:
                    similarity=max((SequenceMatcher(None,item['Item'].lower(),old.lower()).ratio() for old in past),default=0)
                    if similarity>=0.78:
                        item['Suggested action']='Check repeated update'
                        item['Reason']=f'Similar to a prior-month entry (text similarity {similarity:.0%}); ask what changed.'
            result=pd.DataFrame(items)
            if result.empty:st.warning('No reviewable paragraphs or table rows were extracted. Check the document structure.')
            else:
                st.metric('Items extracted for review',len(result))
                st.dataframe(result,use_container_width=True,hide_index=True)
                st.download_button('Download reportability review (CSV)',result.to_csv(index=False).encode('utf-8-sig'),file_name='reportability_review.csv',mime='text/csv')
                st.write('**Monthly interview prompts:** What new grant was submitted/awarded? What enrollment or student outcome changed? Which initiatives reached milestones? What risks or deadlines require leadership attention? What items repeated from last month without new evidence?')
                st.info('For an executive-language rewrite, use your ChatGPT subscription with the approved items and reporting profile. Do not upload protected information without institutional authorization.')
        except Exception as exc:st.error('Could not read Word report: '+str(exc))
    else:st.info('Upload a fictional or de-identified Word draft to review and categorize reportable activities.')
