import pandas as pd
import plotly.express as px
import networkx as nx
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from community import community_louvain

# Load datasets
# Individual level data
discipline_df = pd.read_csv('Discipline_cleaned.csv')
output_publication_df = pd.read_csv('OutputPublication_cleaned.csv')
person_df = pd.read_csv('Person_cleaned.csv')
grant_to_person_df = pd.read_csv('GrantToPerson_cleaned.csv')
grant_df = pd.read_csv("Grant_cleaned.csv")
grant_date_range_df = pd.read_csv('GrantDateRange_cleaned.csv')

# Institutional level data
output_collaboration_df = pd.read_csv('OutputCollaboration_cleaned.csv')
institute_df = pd.read_csv('Institute_cleaned.csv')
person_data_og = pd.read_csv('Person.csv')

# Organizational level data
grant_to_discipline_df = pd.read_csv('GrantToDiscipline_cleaned.csv')
output_award_df = pd.read_csv('OutputAward_cleaned.csv')
output_use_inspired_df = pd.read_csv('OutputUseInspired_cleaned.csv')
grant_to_keyword_df = pd.read_csv('GrantToKeyword_cleaned.csv')
keyword_df = pd.read_csv('Keyword_cleaned.csv')

#Social Network Analysis data
#Using original datasets for customized sna cleaning 
grant_df_original = pd.read_csv("Grant.csv")
grant_to_discipline_df_original= pd.read_csv("GrantToDiscipline.csv")
grant_to_person_df_original = pd.read_csv("GrantToPerson.csv")
person_df_original = pd.read_csv('Person.csv')
institute_df_original = pd.read_csv('Institute.csv')
discipline_df_original = pd.read_csv('Discipline.csv')


# Data cleaning for all levels
grant_df['GrantNumber'] = grant_df['GrantNumber'].str.replace('^G', '', regex=True).astype('int64')
grant_to_person_df['GrantNumber'] = grant_to_person_df['GrantNumber'].str.replace('^G', '', regex=True).astype('int64')
grant_to_discipline_df['GrantNumber'] = grant_to_discipline_df['GrantNumber'].str.replace('^G', '', regex=True).astype('int64')
grant_to_discipline_df['DisciplineNumber'] = grant_to_discipline_df['DisciplineNumber'].str.replace('^D', '', regex=True).astype('int64')
output_publication_df.dropna(subset=['OutputId'], inplace=True)
output_publication_df['GrantNumber'] = output_publication_df['GrantNumber']

# Clean and convert PersonNumber
grant_to_person_df['PersonNumber'] = grant_to_person_df['PersonNumber'].str.replace('^P', '', regex=True).astype('int64')
person_df['PersonNumber'] = person_df['PersonNumber'].str.replace('^P', '', regex=True).astype('int64')

# ======================== Individual Level Analysis ======================== #
# Gender Distribution
grant_person_df = pd.merge(grant_to_person_df, person_df, on='PersonNumber', how='inner')
grant_person_df = pd.merge(grant_person_df, grant_df, on='GrantNumber', how='inner')
grant_person_discipline_df = pd.merge(grant_person_df, discipline_df, left_on='MainDisciplineNumber', right_on='DisciplineNumber', how='left')
top_disciplines = grant_person_discipline_df['Discipline'].value_counts().nlargest(20).index
filtered_df = grant_person_discipline_df[grant_person_discipline_df['Discipline'].isin(top_disciplines)]
gender_discipline_counts_top = filtered_df.groupby(['Discipline', 'Gender']).size().reset_index(name='Count')
fig_gender_distribution = px.bar(gender_discipline_counts_top, x='Discipline', y='Count', color='Gender', title='Gender Distribution Across Top 20 Disciplines', barmode='group')

# Past and future funding success
grant_person_df['GrantStartDate'] = pd.to_datetime(grant_person_df['GrantStartDate'], errors='coerce')
cutoff_date = pd.to_datetime('2000-01-01')

# Filter grants into past and future
past_grants = grant_person_df[grant_person_df['GrantStartDate'] < cutoff_date]
future_grants = grant_person_df[grant_person_df['GrantStartDate'] >= cutoff_date]

# Analyze success rates for top 50 researchers based on grant counts
past_success_counts = past_grants.groupby(['PersonNumber', 'FirstName', 'Surname']).size().reset_index(name='PastGrants')
future_success_counts = future_grants.groupby(['PersonNumber', 'FirstName', 'Surname']).size().reset_index(name='FutureGrants')
success_df = pd.merge(past_success_counts, future_success_counts, on=['PersonNumber', 'FirstName', 'Surname'], how='left').fillna(0)

# Plot for Past vs Future success
fig_success_scatter = px.scatter(success_df.head(50), x='PastGrants', y='FutureGrants', hover_name='FirstName', size='FutureGrants', color='FutureGrants', title='Top 50 Researchers: Past vs Future Funding Success')

# Number of Males and Females Involved in Grants Over Time
grants_by_gender_time = grant_person_df.groupby(['CallDecisionYear', 'Gender']).size().reset_index(name='Count')
fig_gender_time = px.line(grants_by_gender_time, x='CallDecisionYear', y='Count', color='Gender',
                          title='Number of Males and Females Involved in Grants Over Time')

# Ratio of Female to Male in Projects with Only One Team Member
solo_projects = grant_person_df.groupby(['PersonNumber']).size().reset_index(name='Count').query('Count == 1')
solo_gender_ratio = solo_projects.merge(person_df[['PersonNumber', 'Gender']], on='PersonNumber')
gender_ratio = solo_gender_ratio['Gender'].value_counts(normalize=True)
fig_gender_ratio = px.pie(names=gender_ratio.index, values=gender_ratio.values,
                          title='Ratio of Female to Male in Projects with Only One Team Member')

# Number of Solo Male and Female Grant Recipients Over Time
one_team_projects = grant_person_df.groupby('GrantNumber').size().reset_index(name='Count').query('Count == 1')
one_team_gender = one_team_projects.merge(grant_person_df[['GrantNumber', 'PersonNumber']], on='GrantNumber').merge(
    person_df[['PersonNumber', 'Gender']], on='PersonNumber')
one_team_gender_time = one_team_gender.merge(grant_df[['GrantNumber', 'CallDecisionYear']], on='GrantNumber')
small_team_gender_time = one_team_gender_time.groupby(['CallDecisionYear', 'Gender']).size().reset_index(name='Count')
one_small_team_gender_time = px.line(small_team_gender_time, x='CallDecisionYear', y='Count', color='Gender',
                                     title='Number of Solo Male and Female Grant Recipients Over Time')


# Number of Male and Female Grant Recipients Over Time in groups with members <5 and >1
small_team_projects = grant_person_df.groupby('GrantNumber').size().reset_index(name='Count').query('1 < Count < 5')
small_team_gender = small_team_projects.merge(grant_person_df[['GrantNumber', 'PersonNumber']], on='GrantNumber').merge(
    person_df[['PersonNumber', 'Gender']], on='PersonNumber')
small_team_gender_time = small_team_gender.merge(grant_df[['GrantNumber', 'CallDecisionYear']], on='GrantNumber')
small_team_gender_time = small_team_gender_time.groupby(['CallDecisionYear', 'Gender']).size().reset_index(name='Count')
fig_small_team_gender_time = px.line(small_team_gender_time, x='CallDecisionYear', y='Count', color='Gender',
                                     title='Number of Male and Female Grant Recipients Over Time in groups with members <5 and >1')

# Number of Male and Female Grant Recipients Over Time in groups with members >5
large_team_projects = grant_person_df.groupby('GrantNumber').size().reset_index(name='Count').query('Count > 5')
large_team_gender = large_team_projects.merge(grant_person_df[['GrantNumber', 'PersonNumber']], on='GrantNumber').merge(
    person_df[['PersonNumber', 'Gender']], on='PersonNumber')
large_team_gender_time = large_team_gender.merge(grant_df[['GrantNumber', 'CallDecisionYear']], on='GrantNumber')
large_team_gender_time = large_team_gender_time.groupby(['CallDecisionYear', 'Gender']).size().reset_index(name='Count')
fig_large_team_gender_time = px.line(large_team_gender_time, x='CallDecisionYear', y='Count', color='Gender',
                                     title='Number of Male and Female Grant Recipients in Groups of More than 5 Members')


#Top Researcher by number of grant count

top_persons_by_grant_count = grant_to_person_df.groupby('PersonNumber').size().reset_index(name='GrantCount')
top_10_persons = top_persons_by_grant_count.sort_values('GrantCount', ascending=False).head(10)
top_10_persons_details = top_10_persons.merge(person_df[['PersonNumber', 'FirstName']], on='PersonNumber', how='left')
fig_top_10_researcher = px.bar(top_10_persons_details, x='FirstName', y='GrantCount', title='Top Researcher by number of grant count', barmode='group')

print("individual level diagram complete")

# ======================== Institutional Level Analysis ======================== #

# Researcher and Institution graphs
# 1. Top 10 Persons by Number of Grants
top_person_grants = grant_to_person_df['PersonNumber'].value_counts().nlargest(10).reset_index()
top_person_grants.columns = ['PersonNumber', 'GrantCount']
top_person_grants = pd.merge(top_person_grants, person_df, on='PersonNumber', how='left')
fig_top_persons = px.bar(top_person_grants, x='FirstName', y='GrantCount', title="Top 10 Persons by Number of Grants")

# 2. Number of Different Research Institutes Worked at by Top Researchers
# Group by PersonNumber to count the number of times each person received a grant
top_persons_by_grant_count = grant_to_person_df.groupby('PersonNumber').size().reset_index(name='GrantCount')

# Sort by the number of grants and select the top 10 persons
top_10_persons = top_persons_by_grant_count.sort_values('GrantCount', ascending=False).head(10)

# Use the correct column names to merge with person data
top_10_persons_details = top_10_persons.merge(person_df[['PersonNumber', 'FirstName']], on='PersonNumber', how='left')
# Step 1: Filter GrantToPerson for top 10 researchers
top_researchers_grants = grant_to_person_df[grant_to_person_df['PersonNumber'].isin(top_10_persons['PersonNumber'])]

# Step 2: Merge with Grant.csv to get the research institute associated with each grant
top_researchers_institutes = top_researchers_grants.merge(grant_df[['GrantNumber', 'ResearchInstitution']], on='GrantNumber', how='left')

# Step 3: Display the research institutes where each top researcher worked
# Merge with the top 10 persons to include their names
top_researchers_with_institutes = top_researchers_institutes.merge(person_df[['PersonNumber', 'FirstName']], on='PersonNumber', how='left')

# Display the final result (PersonNumber, FirstName, GrantNumber, ResearchInstitution)
top_researchers_with_institutes[['PersonNumber', 'FirstName', 'GrantNumber', 'ResearchInstitution']].drop_duplicates()


# Step 4: Count the number of unique research institutes each researcher has worked with
research_institute_count = top_researchers_with_institutes.groupby('PersonNumber')['ResearchInstitution'].nunique().reset_index(name='InstituteCount')

# Step 5: Merge back with the person names to display results with the researcher names
top_researchers_institute_summary = research_institute_count.merge(person_df[['PersonNumber', 'FirstName']], on='PersonNumber', how='left')
# Step 4: Group by PersonNumber and count the number of unique research institutes each researcher worked at
institute_count_by_person = top_researchers_with_institutes.groupby('PersonNumber')['ResearchInstitution'].nunique().reset_index(name='InstituteCount')

# Step 5: Merge with Person's FirstName for readability
institute_count_by_person = institute_count_by_person.merge(person_df[['PersonNumber', 'FirstName']], on='PersonNumber', how='left')

# person_grant_institute_df = pd.merge(grant_to_person_df, person_df[['PersonNumber', 'ResearchInstitut']], on='PersonNumber', how='inner')
# researcher_institutes_count = person_grant_institute_df.groupby('PersonNumber')['Institute'].nunique().reset_index()
# researcher_institutes_count.columns = ['PersonNumber', 'NumInstitutes']
# top_10_researchers_institutes = researcher_institutes_count.nlargest(10, 'NumInstitutes')
# top_10_researchers_institutes = pd.merge(top_10_researchers_institutes, person_df[['PersonNumber', 'FirstName', 'Surname']], on='PersonNumber', how='left')
# top_10_researchers_institutes['FullName'] = top_10_researchers_institutes['FirstName'] + ' ' + top_10_researchers_institutes['Surname']
fig_research_institutes = px.bar(institute_count_by_person, x='InstituteCount', y='FirstName',
                                 title='Number of Different Research Institutes Worked at by Top Researchers',
                                 labels={'NumInstitutes': 'Number of Institutes', 'FullName': 'Researcher'},
                                 text='InstituteCount')
fig_research_institutes.update_layout(xaxis_title='Researcher', yaxis_title='Number of Institutes', xaxis_tickangle=-45)
fig_research_institutes.update_traces(texttemplate='%{text}', textposition='outside')

# 3. Top Researchers' Grant Affiliations with Top Research Institutes (Heatmap)

top_persons_by_grant_count = grant_to_person_df.groupby('PersonNumber').size().reset_index(name='GrantCount')
top_10_persons = top_persons_by_grant_count.sort_values('GrantCount', ascending=False).head(10)
top_10_persons_details = top_10_persons.merge(person_df[['PersonNumber', 'FirstName']], on='PersonNumber', how='left')
grants_per_institute = grant_df.dropna(subset=['ResearchInstitution']).groupby('ResearchInstitution')['GrantNumber'].count().reset_index(name='GrantCount')
top_institutes = grants_per_institute.sort_values('GrantCount', ascending=False).head(10)
top_researchers_grants = grant_to_person_df[grant_to_person_df['PersonNumber'].isin(top_10_persons['PersonNumber'])]
top_researchers_institutes = top_researchers_grants.merge(grant_df[['GrantNumber', 'ResearchInstitution']], on='GrantNumber', how='left')
top_researchers_institutes = top_researchers_institutes.dropna(subset=['ResearchInstitution'])
top_researchers_at_top_institutes = top_researchers_institutes[top_researchers_institutes['ResearchInstitution'].isin(top_institutes['ResearchInstitution'])]
researcher_institute_affiliation = top_researchers_at_top_institutes.groupby(['PersonNumber', 'ResearchInstitution']).size().reset_index(name='GrantCount')
researcher_institute_affiliation = researcher_institute_affiliation.merge(person_df[['PersonNumber', 'FirstName']], on='PersonNumber', how='left')
filtered_researcher_institute_affiliation = researcher_institute_affiliation[
    (researcher_institute_affiliation['ResearchInstitution'] != 'Unknown')
]
pivot_data = filtered_researcher_institute_affiliation.pivot_table(
    index='ResearchInstitution', 
    columns='PersonNumber', 
    values='GrantCount', 
    aggfunc='sum',
    fill_value=0
)
researcher_names = person_df[person_df['PersonNumber'].isin(pivot_data.columns)].set_index('PersonNumber')['FirstName']
fig_heatmap = go.Figure(data=go.Heatmap(
    z=pivot_data.values,
    x=[researcher_names.get(col, col) for col in pivot_data.columns],  
    y=pivot_data.index,
    colorscale='YlGnBu',  
    hovertemplate='Researcher: %{x}<br>Institution: %{y}<br>Grant Count: %{z}<extra></extra>',
    colorbar=dict(title='Grant Count')
))
for i in range(len(pivot_data.index)):
    for j in range(len(pivot_data.columns)):
        fig_heatmap.add_annotation(
            text=str(pivot_data.values[i][j]),
            x=researcher_names.get(pivot_data.columns[j], pivot_data.columns[j]),
            y=pivot_data.index[i],
            showarrow=False,
            font=dict(color='white' if pivot_data.values[i][j] > pivot_data.values.max() / 2 else 'black')  # Conditional color
        )
fig_heatmap.update_layout(
    title='Top Researchers\' Grant Affiliations with Top Research Institutes',
    xaxis_title='Researcher Name',
    yaxis_title='Research Institution',
    height=600,
    xaxis={'tickangle': 45},
    yaxis=dict(tickmode='linear'), 
    margin=dict(l=80, r=80, t=100, b=80) 
)


# Number of Grants per Year for Top 5 Research Institutions
top_institutions = institute_df['Institute'].value_counts().nlargest(5).index
grants_per_year_institution = grant_df[grant_df['Institute'].isin(top_institutions)].groupby(['CallDecisionYear', 'Institute']).size().reset_index(name='GrantCount')
fig_grants_per_year_institution = px.line(grants_per_year_institution, x='CallDecisionYear', y='GrantCount', color='Institute',
                                          title='Number of Grants per Year for Top 5 Research Institutions')

# Total Grant Amount per Year for Top 5 Research Institutions (in Billions)
grant_amount_per_year_institution = grant_df[grant_df['Institute'].isin(top_institutions)].groupby(['CallDecisionYear', 'Institute'])['AmountGrantedAllSets'].sum().reset_index()
grant_amount_per_year_institution['AmountGrantedAllSets'] = grant_amount_per_year_institution['AmountGrantedAllSets'] / 1e9  # Convert to billions
fig_grant_amount_per_year_institution = px.line(grant_amount_per_year_institution, x='CallDecisionYear', y='AmountGrantedAllSets', color='Institute',
                                                title='Total Grant Amount per Year for Top 5 Research Institutions (in Billions)')

# Total Grant Amount by Research Institution (in Billions)
total_grant_amount_institution = grant_df.groupby('Institute')['AmountGrantedAllSets'].sum().reset_index()
total_grant_amount_institution['AmountGrantedAllSets'] = total_grant_amount_institution['AmountGrantedAllSets'] / 1e9  # Convert to billions
fig_total_grant_amount_institution = px.bar(total_grant_amount_institution.nlargest(10, 'AmountGrantedAllSets'), x='Institute', y='AmountGrantedAllSets',
                                            title='Total Grant Amount by Research Institution (in Billions)')
print("Institutional level diagram complete")

# ======================== Organizational Level Analysis ======================== #

# Top 10 Countries Based on Total Amount of Grants Received
grants_by_country = grant_df.groupby('InstituteCountry')['AmountGrantedAllSets'].sum().reset_index()
grants_by_country = grants_by_country.nlargest(10, 'AmountGrantedAllSets')
fig_top_countries_grants = px.bar(grants_by_country, x='InstituteCountry', y='AmountGrantedAllSets',
                                  title="Top 10 Countries Based on Total Amount of Grants Received",
                                  labels={'AmountGrantedAllSets': 'Total Amount (CHF)', 'InstituteCountry': 'Country'})

# Top 10 Disciplines Based on Number of Grants Received
grants_by_discipline = grant_to_discipline_df.groupby('DisciplineNumber').size().reset_index(name='GrantCount')
grants_by_discipline = grants_by_discipline.nlargest(10, 'GrantCount')
grants_by_discipline = pd.merge(grants_by_discipline, discipline_df[['DisciplineNumber', 'Discipline']], on='DisciplineNumber', how='left')
fig_top_disciplines_grants = px.bar(grants_by_discipline, x='Discipline', y='GrantCount',
                                    title="Top 10 Disciplines Based on Number of Grants Received",
                                    labels={'GrantCount': 'Number of Grants', 'Discipline': 'Discipline'})

# Assume 'Keywords' is a column in grant_df where keywords are stored as a comma-separated string
# grant_df['Keywords'] = grant_df['Keywords'].fillna('')  # Handle missing data
# keywords_series = grant_df['Keywords'].str.split(',', expand=True).stack().value_counts().nlargest(10)
# keywords_df = pd.DataFrame(keywords_series).reset_index()
# keywords_df.columns = ['Keyword', 'GrantCount']

# fig_top_keywords_grants = px.bar(keywords_df, x='Keyword', y='GrantCount',
#                                 title="Top 10 Keywords Based on Number of Grants Received",
#                                 labels={'GrantCount': 'Number of Grants', 'Keyword': 'Keyword'})

# Top 10 Countries Based on Total Amount of Grants Received (in Billions)
grants_by_country_billion = grants_by_country.copy()
grants_by_country_billion['AmountGrantedAllSets'] = grants_by_country_billion['AmountGrantedAllSets'] / 1e9  # Convert to billions
fig_top_countries_grants_billion = px.bar(grants_by_country_billion, x='InstituteCountry', y='AmountGrantedAllSets',
                                          title="Top 10 Countries Based on Total Amount of Grants Received (in Billions)",
                                          labels={'AmountGrantedAllSets': 'Total Amount (Billion CHF)', 'InstituteCountry': 'Country'})


# Identify the top 5 keywords
# Group by KeywordId and count the number of grants
top_keywords = grant_to_keyword_df.groupby('KeywordId').size().sort_values(ascending=False).head(10)

# Identify the top 5 keywords by grant count again
top_5_keywords = top_keywords.head(5)

# Fix GrantNumber format in the grant_to_keyword dataframe
# grant_to_keyword_df['GrantNumber'] = 'G' + grant_to_keyword_df['GrantNumber'].astype(str)

# Merge grant data with keyword data to get the CallDecisionYear and grant amounts
grant_keyword_merged_df = pd.merge(grant_df[['GrantNumber', 'CallDecisionYear', 'AmountGrantedAllSets']],
                                   grant_to_keyword_df,
                                   on='GrantNumber',
                                   how='left')

# Merge with keyword names
grant_keyword_merged_df = pd.merge(grant_keyword_merged_df, keyword_df, left_on='KeywordId', right_on='Id', how='left')

# Filter for the top 5 keywords
top_5_keyword_funding_df = grant_keyword_merged_df[grant_keyword_merged_df['KeywordId'].isin(top_5_keywords.index)]

# Group by year and keyword, count occurrences, and sum the grant amounts
keyword_year_count = top_5_keyword_funding_df.groupby(['CallDecisionYear', 'Word']).size().reset_index(name='Count')
keyword_year_sum = top_5_keyword_funding_df.groupby(['CallDecisionYear', 'Word'])['AmountGrantedAllSets'].sum()
# Reset the index of the Series so that 'CallDecisionYear' and 'Word' become columns in a DataFrame
keyword_year_sum_reset = keyword_year_sum.reset_index()

# Prepare data for plotting (in billions)
keyword_year_sum_billion = keyword_year_sum / 1e9
# Combine count and sum for clarity if needed
time_series_df = keyword_year_sum_billion.unstack()

fig_keyword_top_5_time_series = px.line(keyword_year_count, x='CallDecisionYear', y='Count', color='Word',
                          title='Top 5 keyword over the time based on number of grant')
fig_keyword_top_5_time_series_for_funding = px.line(keyword_year_sum_reset, x='CallDecisionYear', y='AmountGrantedAllSets', color='Word',
                          title='Top 5 keyword over the time and the grant received by them')


print("Organizational diagram complete")

# ======================Social Network Analysis ===================== #


print("preparing for social network analysis")
# cusomized data cleaning for generating graphs
grant_to_discipline_df_original['DisciplineNumber'] = grant_to_discipline_df_original['DisciplineNumber'].str.replace('^D', '', regex=True).astype('int64')

#mergind datasets for generating bipartite network
grant_filtered = grant_df_original[grant_df_original['CallDecisionYear'].between(2019, 2022)]
top_disciplines = grant_filtered['MainDiscipline'].value_counts().nlargest(5).index
grant_filtered = grant_filtered[grant_filtered['MainDiscipline'].isin(top_disciplines)]
merged_data_filtered = pd.merge(grant_to_discipline_df_original, discipline_df_original, on='DisciplineNumber')
merged_data_filtered = merged_data_filtered[merged_data_filtered['GrantNumber'].isin(grant_filtered['GrantNumber'])]
top_institutes = grant_filtered['Institute'].value_counts().nlargest(5).index.tolist()


#Grants vs. Persons, Persons vs. Institutes and Grants vs. Disciplines bipartite graph
top_institutes_data = institute_df_original[institute_df_original['InstituteNumber'].isin(top_institutes)]
# Grants vs. Persons
G_grants_persons = nx.Graph()
valid_grants = set(grant_filtered['GrantNumber'])

# Use a single loop to add nodes and edges
for _, row in grant_to_person_df_original[grant_to_person_df_original['GrantNumber'].isin(valid_grants)].iterrows():
    G_grants_persons.add_node(row['GrantNumber'], bipartite=0)  # Grant node
    G_grants_persons.add_node(row['PersonNumber'], bipartite=1)  # Person node
    G_grants_persons.add_edge(row['GrantNumber'], row['PersonNumber'])  # Connect grant to person

# Persons vs. Institutes
G_persons_institutes = nx.Graph()
grant_to_institute = grant_filtered[['GrantNumber', 'Institute']].dropna()

# Pre-filtered DataFrame for efficiency
filtered_grant_to_institute = grant_to_institute[grant_to_institute['Institute'].isin(top_institutes)]

for _, row in filtered_grant_to_institute.iterrows():
    G_persons_institutes.add_node(row['Institute'], bipartite=0)  # Institute node
    persons = grant_to_person_df_original[grant_to_person_df_original['GrantNumber'] == row['GrantNumber']]['PersonNumber']
    
    # Add nodes and edges for persons in one go
    for person in persons:
        G_persons_institutes.add_node(person, bipartite=1)  # Person node
        G_persons_institutes.add_edge(row['Institute'], person)  # Connect institute to person

# Grants vs. Disciplines
G_grants_disciplines = nx.Graph()

for _, row in merged_data_filtered.iterrows():
    G_grants_disciplines.add_node(row['GrantNumber'], bipartite=0)  # Grant node
    G_grants_disciplines.add_node(row['Discipline'], bipartite=1)  # Discipline node
    G_grants_disciplines.add_edge(row['GrantNumber'], row['Discipline'])  # Connect grant to discipline

def plot_graph(G, title):
    pos = nx.spring_layout(G)  # Layout for 2D
    edge_x = []
    edge_y = []
    
    # Preallocate lists for edges and nodes
    edge_count = G.number_of_edges()
    edge_x = [None] * (edge_count * 3)  # Each edge contributes 3 points
    edge_y = [None] * (edge_count * 3)
    
    # Fill edge coordinates
    idx = 0
    for x0, y0, x1, y1 in ((pos[edge[0]][0], pos[edge[0]][1], pos[edge[1]][0], pos[edge[1]][1]) for edge in G.edges()):
        edge_x[idx:idx + 3] = [x0, x1, None]
        edge_y[idx:idx + 3] = [y0, y1, None]
        idx += 3

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []

    # Use a single loop to gather node data
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        # Use a ternary expression to determine color
        color = 'red' if G.nodes[node].get('bipartite') == 0 else 'blue'
        node_color.append(color)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(showscale=False, size=10, color=node_color, line_width=2)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         title=title,
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=0, l=0, r=0, t=40),
                         xaxis=dict(showgrid=False, zeroline=False),
                         yaxis=dict(showgrid=False, zeroline=False)
                     ))
    
    return fig

grant_vs_person=plot_graph(G_grants_persons, 'Grants vs. Persons')
print("Grants vs. Persons bipartite network complete")
person_vs_institute=plot_graph(G_persons_institutes, 'Persons vs. Institutes')
print("Persons vs. Institutes bipartite network complete")
grant_vs_discipline=plot_graph(G_grants_disciplines, 'Grants vs. Disciplines')
print("Grants vs. Disciplines bipartite network complete")



# 3D Grant-Discipline-Person-Institute Bipartite Network (2019-2022)
B = nx.Graph()
for _, row in merged_data_filtered.iterrows():
    B.add_node(row['GrantNumber'], bipartite=0)  # Grant node
    B.add_node(row['Discipline'], bipartite=1)  # Discipline node
    B.add_edge(row['GrantNumber'], row['Discipline'])  # Edge between grant and discipline

grant_to_institute = grant_filtered[['GrantNumber', 'Institute']].dropna()
for _, row in grant_to_institute.iterrows():
    if row['Institute'] in top_institutes:
        B.add_node(row['Institute'], bipartite=1)  # Institute node
        B.add_edge(row['GrantNumber'], row['Institute'])  # Edge between grant and institute

grant_to_person_filtered = grant_to_person_df_original[grant_to_person_df_original['GrantNumber'].isin(grant_filtered['GrantNumber'])]
for _, row in grant_to_person_filtered.iterrows():
    B.add_node(row['PersonNumber'], bipartite=1)  # Person node
    B.add_edge(row['GrantNumber'], row['PersonNumber'])  # Edge between grant and person

pos = {node: (np.random.rand(), np.random.rand(), np.random.rand()) for node in B.nodes()}

edges = np.array(list(B.edges))
edge_x = np.ravel([[pos[edge[0]][0], pos[edge[1]][0], None] for edge in edges])
edge_y = np.ravel([[pos[edge[0]][1], pos[edge[1]][1], None] for edge in edges])
edge_z = np.ravel([[pos[edge[0]][2], pos[edge[1]][2], None] for edge in edges])

edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)
node_x = [pos[node][0] for node in B.nodes()]
node_y = [pos[node][1] for node in B.nodes()]
node_z = [pos[node][2] for node in B.nodes()]
node_text = list(B.nodes())
node_color = [
    'red' if node in merged_data_filtered['GrantNumber'].unique() else
    'green' if node in merged_data_filtered['Discipline'].unique() else
    'blue' if node in top_institutes else
    'orange'  
    for node in B.nodes()
]
node_trace = go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers',
    hoverinfo='text',
    text=node_text,
    marker=dict(showscale=False, size=10, color=node_color, line_width=2)
)
multilayered_graph = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                scene=dict(
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False),
                    zaxis=dict(showgrid=False)
                ),
                annotations=[dict(
                    text="Red: Grants, Green: Disciplines, Blue: Institutes, Orange: People",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )]
             )
)
print("3d multi layer network complete")


#Multi-level bipartite graph top centrality plots 
degree_centrality = nx.degree_centrality(B)
betweenness_centrality = nx.betweenness_centrality(B)
closeness_centrality = nx.closeness_centrality(B)
centrality_df = pd.DataFrame({
    'Node': degree_centrality.keys(),
    'Degree Centrality': degree_centrality.values(),
    'Betweenness Centrality': betweenness_centrality.values(),
    'Closeness Centrality': closeness_centrality.values()
})
top_degree = centrality_df.nlargest(5, 'Degree Centrality')
top_node_list = top_degree['Node'].tolist()
top_betweenness = centrality_df.nlargest(5, 'Betweenness Centrality')
top_closeness = centrality_df.nlargest(5, 'Closeness Centrality')
# Create bar graph for Degree Centrality
fig_degree = go.Figure()
fig_degree.add_trace(go.Bar(
    x=top_degree['Node'],
    y=top_degree['Degree Centrality'],
    name='Degree Centrality',
    marker_color='blue'
))
# Create bar graph for Betweenness Centrality
fig_betweenness = go.Figure()
fig_betweenness.add_trace(go.Bar(
    x=top_betweenness['Node'],
    y=top_betweenness['Betweenness Centrality'],
    name='Betweenness Centrality',
    marker_color='green'
))
# Create bar graph for Closeness Centrality
fig_closeness = go.Figure()
fig_closeness.add_trace(go.Bar(
    x=top_closeness['Node'],
    y=top_closeness['Closeness Centrality'],
    name='Closeness Centrality',
    marker_color='orange'
))
fig_degree.update_layout(
    xaxis_title='Nodes',
    yaxis_title='Degree Centrality',
    template='plotly_white'
)
fig_betweenness.update_layout(
    xaxis_title='Nodes',
    yaxis_title='Betweenness Centrality',
    template='plotly_white'
)
fig_closeness.update_layout(
    xaxis_title='Nodes',
    yaxis_title='Closeness Centrality',
    template='plotly_white'
)
print("3d multi layer network analysis complete")

 # Perform community detection using the Louvain method
partition = community_louvain.best_partition(B)

# Create a DataFrame to count the number of nodes in each community
community_counts = pd.Series(partition).value_counts().reset_index()
community_counts.columns = ['Community', 'Node Count']

top_communities = community_counts.nlargest(5, 'Node Count')
community_detection = go.Figure()
community_detection.add_trace(go.Bar(
    x=top_communities['Community'].astype(str),  # Convert to string for better labeling
    y=top_communities['Node Count'],
    marker_color='purple'
))
community_detection.update_layout(
    # title='Top Communities Detected by Louvain Method',
    xaxis_title='Community',
    yaxis_title='Number of Nodes',
    template='plotly_white'
)


#SNSF Historical Funding Allocation For Disciplines With Top Centralities
grant_df_original['GrantStartDate'] = pd.to_datetime(grant_df_original['GrantStartDate'])
merged_df = grant_df_original.merge(discipline_df, left_on='MainDiscipline', right_on='Discipline', how='inner')
merged_df['Year'] = merged_df['GrantStartDate'].dt.year
# Group by Year and Discipline, summing the Amount Granted
funding_trends = merged_df.groupby(['Year', 'Discipline'])['AmountGrantedAllSets'].sum().reset_index()
filtered_funding_trends = funding_trends[funding_trends['Discipline'].isin(top_node_list)]
# Create an interactive line plot for top 5 disciplines
grant_history = px.line(filtered_funding_trends,
              x='Year',
              y='AmountGrantedAllSets',
              color='Discipline',
              labels={'AmountGrantedAllSets': 'Total Funding Amount (CHF)', 'Year': 'Year'},
              markers=True)
print("historical funding allocation for disciplines with top centralities complete")


# ======================== Dash App Layout ======================== #
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Researcher Dashboard", style={'textAlign': 'center'}),

    # Tabs for different levels
    dbc.Row([
        dbc.Col(dcc.Tabs(id='tabs', value='individual', children=[
            dcc.Tab(label='Individual Level', value='individual'),
            dcc.Tab(label='Institutional Level', value='institutional'),
            dcc.Tab(label='Organizational Level', value='organizational'),
            dcc.Tab(label='Social Network Analysis', value='sna')
        ]))
    ]),

    # Content will be updated based on the selected level
    html.Div(id='level-content')
])

# ======================== Callback for Tabs ======================== #
@app.callback(
    Output('level-content', 'children'),
    [Input('tabs', 'value')]
)
def render_content(tab):
    if tab == 'individual':
        return html.Div([
            html.H3('Gender Distribution Across Disciplines'),
            dcc.Graph(figure=fig_gender_distribution),
            html.H3('Top 50 Researchers: Past vs Future Funding Success'),
            dcc.Graph(figure=fig_success_scatter),
            html.H3('Number of Males and Females Involved in Grants Over Time'),
            dcc.Graph(figure=fig_gender_time),
            html.H3('Ratio of Female to Male in Projects with Only One Team Member'),
            dcc.Graph(figure=fig_gender_ratio),
            html.H3('Number of Solo Male and Female Grant Recipients Over Time'),
            dcc.Graph(figure=one_small_team_gender_time),
            html.H3('Number of Male and Female Grant Recipients Over Time in groups with members <5 and >1'),
            dcc.Graph(figure=fig_small_team_gender_time),
            html.H3('Number of Male and Female Grant Recipients in Groups of More than 5 Members'),
            dcc.Graph(figure=fig_large_team_gender_time),
            html.H3('Top 10 researcher based on number of grant they are involved'),
            dcc.Graph(figure=fig_top_10_researcher),
        ])
    elif tab == 'institutional':
        return html.Div([
            html.H3('Number of Different Research Institutes Worked at by Top Researchers'),
            dcc.Graph(figure=fig_research_institutes),
            html.H3('Top Researchers\' Grant Affiliations with Top Research Institutes (Heatmap)'),
            dcc.Graph(figure=fig_heatmap),
            html.H3('Number of Grants per Year for Top 5 Research Institutions'),
            dcc.Graph(figure=fig_grants_per_year_institution),
            html.H3('Total Grant Amount per Year for Top 5 Research Institutions (in Billions)'),
            dcc.Graph(figure=fig_grant_amount_per_year_institution),
            html.H3('Total Grant Amount by Research Institution (in Billions)'),
            dcc.Graph(figure=fig_total_grant_amount_institution)
        ])
    elif tab == 'organizational':
        return html.Div([
            # html.H3('Organizational Innovation Outputs by Discipline'),
            # dcc.Graph(figure=fig_organizational_innovation),
            html.H3('Top 10 Countries Based on Total Amount of Grants Received'),
            dcc.Graph(figure=fig_top_countries_grants),
            html.H3('Top 10 Disciplines Based on Number of Grants Received'),
            dcc.Graph(figure=fig_top_disciplines_grants),
            html.H3('Top 5 Keywords Based on Number of Grants Received'),
            dcc.Graph(figure=fig_keyword_top_5_time_series),
            html.H3('Top 5 keyword over the time and the grant received by them'),
            dcc.Graph(figure=fig_keyword_top_5_time_series_for_funding),
            html.H3('Top 10 Countries Based on Total Amount of Grants Received (in Billions)'),
            dcc.Graph(figure=fig_top_countries_grants_billion)
        ])
    elif tab == 'sna':
        return html.Div([
            html.H3('2D Grant vs. Person Bipartite Network Sampled (2019-2022)'),
            dcc.Graph(
                figure=grant_vs_person,
                style={'height': '700px'}
                ),
            html.P(
                    """
                    Communities: 1012 detected, Average Path Length: Not applicable (graph is disconnected)
                    """,
                    style={'fontSize': '16px', 'padding': '10px 0px 60px', 'font-weight': '600'}
                ),
            html.H3('2D Person vs. Institute Bipartite Network Sampled (2019-2022)'),
            dcc.Graph(
                figure=person_vs_institute,
                style={'height': '700px'}
                ),
            html.P(
                    """
                    Communities: 5 detected, Average Path Length: Not applicable (graph is disconnected)
                    """,
                    style={'fontSize': '16px', 'padding': '10px 0px 60px', 'font-weight': '600'}
                ),
            html.H3('2D Grant vs. Discipline Bipartite Network Sampled (2019-2022)'),
            dcc.Graph(
                figure=grant_vs_discipline,
                style={'height': '700px'}
                ),
            html.P(
                    """
                    Communities: 6 detected, Average Path Length: 4.0592
                    """,
                    style={'fontSize': '16px', 'padding': '10px 0px 60px', 'font-weight': '600'}
                ),            
            html.H3('3D Grant-Discipline-Person-Institute Bipartite Network Sampled (2019-2022)'),
            dcc.Graph(
                figure=multilayered_graph,
                style={'height': '700px'}
                ),
            html.P(
                    """
                    Communities: 138 detected, Number of Nodes: 4569, Average Path Length: 5.2237
                    """,
                    style={'fontSize': '16px', 'padding': '10px 0px 60px', 'font-weight': '600'}
                ),    
            html.H4("""
                    Top 5 Degree Centralities
                    """,
                    style={'padding': '40px 0px 0px'}
                    ),
            dcc.Graph(
                figure=fig_degree,
                style={'height': '600px'}
                ),
            html.P(
                    """
                    Top Node: Psychology (degree of centrality: 0.087033)
                    """,
                    style={'fontSize': '16px', 'padding': '10px 0px 60px', 'font-weight': '600'}
                ),
            html.H4(
                    """
                    Top 5 Betweenness Centralities
                    """,
                    style={'padding': '40px 0px 0px'}
                    ),
            dcc.Graph(
                figure=fig_betweenness,
                style={'height': '600px'}
                ),
            html.P(
                    """
                    Top Node: Psychology (betweenness centrality: 0.463613)
                    """,
                    style={'fontSize': '16px', 'padding': '10px 0px 60px', 'font-weight': '600'}
                ),
            html.H4(
                    """
                    Top 5 Closeness Centralities
                    """,
                    style={'padding': '40px 0px 0px'}
                    ),
            dcc.Graph(
                figure=fig_closeness,
                style={'height': '600px'}
                ),
            html.P(
                    """
                    Top Node: G190761 (most influential grant; closeness centrality: 0.292934)
                    """,
                    style={'fontSize': '16px', 'padding': '10px 0px 60px', 'font-weight': '600'}
                ),
            html.H4(
                    """
                    Top 5 Communities Detected by Louvain Method
                    """,
                    style={'padding': '40px 0px 0px'}
                    ),
            dcc.Graph(
                figure=community_detection,
                style={'height': '600px'}
                ),
            html.P(
                    """
                    High node count indicates potential complexity in community structure
                    """,
                    style={'fontSize': '16px', 'padding': '10px 0px 60px', 'font-weight': '600'}
                ),
            html.H4(
                    """
                    SNSF Historical Funding Allocation For Disciplines With Top Centralities
                    """,
                    style={'padding': '40px 0px 0px'}
                    ),
            dcc.Graph(
                figure=grant_history,
                style={'height': '600px'}
                ),
            html.P(
                    """
                    Molecular Biology (CHF 775,305,741); Neurophysiology and Brain Research (CHF 803,151,199)                    
                    """,
                    style={'fontSize': '16px', 'padding': '10px 0px 60px', 'font-weight': '600'}
                )
        ])
    else:
        return html.Div("Please select a valid level.")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)

           

