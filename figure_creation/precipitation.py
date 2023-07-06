import pandas as pd
import matplotlib.pyplot as plt


def load_data(in_path):
    df = pd.read_csv(in_path, sep=';')
    df['prec'] = pd.to_numeric(df['prec'].str.replace(',','.'))
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    return df

def filter_dates(df, start, end):
    df_start = df[df['date'] >= start]
    df_filtered = df_start[df_start['date'] <= end]
    return df_filtered

def plot_prec(df, ax):
    ax.bar(df['date'], df['prec'])
    ax.set_ylim([0, 35])
    ax.set_ylabel('Precipitation [mm]')

def plot(df1, df2):
    fig, (ax1, ax2) = plt.subplots(2)

    plot_prec(df1, ax1)
    plot_prec(df2, ax2)

    plt.show()


if __name__ == '__main__':
    in_data = r'C:\Users\dd\Documents\NATUR_CUNI\_dp\precipitation\P2VYSK01_SRA_N.csv'
    df_in = load_data(in_data)

    df_2021 = filter_dates(df_in, '2021-04-01', '2021-09-30')
    df_2022 = filter_dates(df_in, '2022-04-01', '2022-09-30')

    #plot_prec(df_2021)
    #plot_prec(df_2022)
    plot(df_2021, df_2022)
