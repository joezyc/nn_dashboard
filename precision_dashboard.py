import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import date

today = str(date.today())
data = pd.read_csv('../storage/outputs/error_table/error_all_currencies.csv')

nn_data = pd.read_csv('../storage/outputs/bond_activity/bond_activity_result.csv')

def nn_dashboards(data):
    import plotly.express as px
    data = data.dropna(how = 'any')
    data = data.rename(columns = {'issuer_name':'Issuer','issuer_id': 'Issuer ID','currency':'Currency','sector':'Sector','nn_name':'Nearest Neighbour',
                           'nn_id':'NN ID','mean_ratio':'Mean Ratio','count':'Number of Securities'})
    fig = px.treemap(data, path = ['Currency','Sector','Nearest Neighbour','Issuer ID'], values = 'Number of Securities',color = 'Sector',
                     hover_name = 'Issuer',
                     hover_data = ['Issuer','Issuer ID','Currency','Number of Securities','Sector','Nearest Neighbour','NN ID','Mean Ratio'],
                     title = 'Nearest Neighbour Dashboard')
    fig.show()
    fig.write_html("../storage/outputs/bond_activity/nn_dashboard.html")
    
def precision_by_sectors(currency, data):
    df = data[data['currency'] == currency]
    trade_date = df['trade_date'].dropna().unique().tolist()[0]

    series_sectors = pd.Series(df['sector_name'].dropna().unique())

    df_by_sector = pd.DataFrame()
    df_by_sector['sector'] = series_sectors
    series_avg_error = df.groupby(['sector_name'])['error'].mean()
    series_count_issuer = df.groupby(['sector_name'])['name'].nunique()
    df_by_sector = df_by_sector.merge(series_avg_error, left_on = 'sector', right_index = True, how = 'left')
    df_by_sector = df_by_sector.merge(series_count_issuer, left_on='sector', right_index=True, how='left')

    df_by_sector = df_by_sector.rename(columns = {'error' : 'avg_error','name' : 'count_issuer'})

    hover_text = []
    bubble_size = []

    for index, row in df_by_sector.iterrows():
        hover_text.append(('Currency:{cur}<br>' +
                           'Sector:{sector}<br>' +
                           'Avg. Error:{error}<br>' +
                           'Count of issuers:{issuer}<br>').format(cur = currency,
                                                                  sector = row['sector'],
                                                                  error = round(row['avg_error'],3),
                                                                  issuer = row['count_issuer']))
        bubble_size.append(row['count_issuer'])

    df_by_sector['text'] = hover_text
    df_by_sector['size'] = bubble_size

    fig = go.Figure()

    if currency == 'CAD':
        scale = 1.5
    elif currency == 'USD':
        scale = 0.2
    elif currency == 'EUR':
        scale = 0.6
    else:
        scale = 1

    for i in df_by_sector.index:
        fig.add_trace(
            go.Scatter(
            x = [df_by_sector.loc[i,'avg_error']],
            y = [df_by_sector.loc[i,'sector']],
            name = df_by_sector.loc[i,'sector'],
            text = [df_by_sector.loc[i,'text']],
            marker_size = [df_by_sector.loc[i,'size'] * scale]
            )
        )

    fig.update_traces(mode='markers', marker=dict(line_width=2))

    fig.update_layout(
        autosize = True,
        title = f'Precision Dashboard By Sectors '
                f'for {currency} on {trade_date}',
        xaxis = dict(title = 'Average Error', gridcolor = 'white', gridwidth = 2),
        yaxis=dict(title='Sectors', gridcolor='white', gridwidth=2),
        font = dict(size = 8, color = 'Grey'),
        paper_bgcolor = 'rgb(243, 243, 243)',
        plot_bgcolor = 'rgb(243, 243, 243)',
    )

    #fig.show()
    fig.write_html(f"../storage/dashboards/dashboard_by_sector_{currency}_{today}.html")
    #fig.write_image(f"../storage/dashboards//dashboard_by_sector_{currency}.svg")

def precision_by_company(data):
    trade_date = data['trade_date'].dropna().unique().tolist()[0]

    series_company = data['name'].sort_values().unique()

    df_by_company = pd.DataFrame(columns = ['name','currency','issuer_type','avg_error','count_issuer'])

    for company in series_company.tolist():
        df_company = data[data.name == company]
        list_currency = df_company['currency'].unique().tolist()

        for currency in list_currency:
            df_at_cur = df_company[df_company['currency'] == currency]
            issuer_type = df_at_cur['high_issuers'].dropna().unique().tolist()[0]
            if issuer_type == 1:
                issuer_type = 'High'
            else:
                issuer_type = 'Low'
            avg_error = df_at_cur['error'].mean()
            count_issuer = df_at_cur.shape[0]
            df_by_company = pd.concat(
                [df_by_company, pd.DataFrame(np.array([[company, currency, issuer_type, avg_error, count_issuer]]),
                                             columns=['name', 'currency', 'issuer_type', 'avg_error', 'count_issuer'])])

    df_by_company
    hover_text = []
    bubble_size = []

    for index, row in df_by_company.iterrows():
        hover_text.append(('Currency:{cur}<br>' +
                           'Name:{name}<br>' +
                           'Avg. Error:{error}<br>' +
                           'Count of issuers:{issuer}<br>' +
                           'Issuer Type: {issuer_type}<br>').format(cur= row['currency'],
                                                                    name =row['name'],
                                                                    error=round(float(row['avg_error']), 3),
                                                                    issuer=row['count_issuer'],
                                                                    issuer_type = row['issuer_type']))
        bubble_size.append(row['count_issuer'])

    df_by_company['text'] = hover_text
    df_by_company['size'] = bubble_size

    df_by_company = df_by_company.reset_index(drop = True )

    fig = go.Figure()

    for currency in df_by_company['currency'].unique():
        fig.add_trace(
            go.Scatter(
            x = df_by_company.loc[df_by_company['currency'] == currency,'avg_error'].tolist(),
            y = df_by_company.loc[df_by_company['currency'] == currency,'name'].tolist(),
            name = currency,
            text = df_by_company.loc[df_by_company['currency'] == currency,'text'].tolist(),
            marker_size = (df_by_company.loc[df_by_company['currency'] == currency,'size']).apply(func = int).tolist()
            )
        )

    fig.update_traces(mode='markers', marker=dict(line_width=1))

    fig.update_layout(
        title=f'Precision Dashboard By Company on {trade_date}',
        xaxis=dict(title='Average Error', gridcolor='white', gridwidth=1),
        yaxis=dict(title='Company Name', gridcolor='white', gridwidth=1),
        font=dict(size=11, color='Grey'),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        height = 2000,
    )

    fig.update_xaxes(automargin=True)

    #fig.show()
    fig.write_html(f"../storage/dashboards/dashboard_by_company_{today}.html")
    #fig.write_image("../storage/dashboards/dashboard_by_company.svg")

if __name__  == '__main__':
    nn_dashboards(nn_data)