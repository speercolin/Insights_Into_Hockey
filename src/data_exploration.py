import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Importing NHL season data
nhl_data = pd.read_csv('sportsref_download.csv')
nhl_data['Team'] = nhl_data['Team'].str.replace('*', '')
print(nhl_data)

# Creating column variables from data
teams = nhl_data['Team']
team_points = nhl_data['PTS']
goals_for = nhl_data['GF']
goals_against = nhl_data['GA']
wins = nhl_data['W']
losses = nhl_data['L']
overtime_losses = nhl_data['OL']
powerplay_perc = nhl_data['PP%']
penalty_kill_perc = nhl_data['PK%']
age = nhl_data['AvAge']

# Goals For vs Goals Against
def goals_for_vs_goals_against():

    # Plotting league comparison
    plt.figure(figsize=(12, 6))
    plt.scatter(goals_for, goals_against, color='purple', alpha=0.5)

    # Hovering labels
    mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(teams[sel.target.index]))

    plt.title('Goals For vs Goals Against')
    plt.xlabel('Goals For')
    plt.ylabel('Goals Against')
    plt.grid(True)
    plt.show()


def wins_vs_losses():

    # Wins vs Losses
    plt.figure(figsize=(12, 6))
    plt.barh(teams, wins, color='green', label='Wins')
    plt.barh(teams, losses, color='red', label='Losses', left=wins)
    plt.xlabel('Number of Games')
    plt.title('Win-Loss Record for Each Team')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()


def powerplay_vs_penalty_kill():

    # Plotting the correlation between PP% and PK%
    plt.figure(figsize=(10, 6))
    plt.scatter(powerplay_perc, penalty_kill_perc, color='red', alpha=0.5)
    plt.title('Correlation between Powerplay Percentage and Penalty Kill Percentage')
    plt.xlabel('Powerplay Percentage')
    plt.ylabel('Penalty Kill Percentage')
    plt.grid(True)

    # Hovering labels
    mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(teams[sel.target.index]))

    # Calculating the correlation coefficient
    powerplay_penalty_correlation_coefficient = powerplay_perc.corr(penalty_kill_perc)
    plt.text(0.1, 0.9, f'Correlation coefficient: {powerplay_penalty_correlation_coefficient:.2f}', transform=plt.gca().transAxes)
    plt.show()


def powerplay_vs_wins():

    # Plotting the correlation between wins and PP%
    plt.figure(figsize=(10, 6))
    plt.scatter(powerplay_perc, wins, color='purple', alpha=0.5)
    plt.title('Correlation between Powerplay Percentage and Wins')
    plt.xlabel('Powerplay Percentage')
    plt.ylabel('Wins')
    plt.grid(True)

    # Hovering labels
    mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(teams[sel.target.index]))

    # Calculating the correlation coefficient
    powerplay_wins_correlation_coefficient = powerplay_perc.corr(wins)
    plt.text(0.1, 0.9, f'Correlation coefficient: {powerplay_wins_correlation_coefficient:.2f}', transform=plt.gca().transAxes)
    plt.show()


def age_vs_wins():

    # Plotting the correlation between average age and wins
    plt.figure(figsize=(10, 6))
    plt.scatter(age, wins, color='cyan', alpha=0.5)
    plt.title('Correlation between Average Age and Wins')
    plt.xlabel('Average Age')
    plt.ylabel('Wins')
    plt.grid(True)

    # Hovering labels
    mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(teams[sel.target.index]))

    # Calculating the correlation coefficient
    age_wins_correlation_coefficient = age.corr(wins)
    plt.text(0.1, 0.9, f'Correlation coefficient: {age_wins_correlation_coefficient:.2f}', transform=plt.gca().transAxes)
    plt.show()


def optimise_k_means(data, max_k):

    means = []
    interias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        interias.append(kmeans.inertia_)

    # Elbow Plot
    fig = plt.subplots(figsize=(10, 6))
    plt.plot(means, interias, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()


def kmeans_goals_for_against():

    scaler = StandardScaler()
    
    kmeans_data = nhl_data[['W', 'GF', 'GA', 'PP%', 'PK%']]
    kmeans_data[['W', 'GF', 'GA', 'PP%', 'PK%']] == scaler.fit_transform(kmeans_data[['W', 'GF', 'GA', 'PP%', 'PK%']])

    optimise_k_means(kmeans_data[['PP%', 'PK%']], 10)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(kmeans_data[['PP%', 'PK%']])

    kmeans_data['kmeans_3'] = kmeans.labels_
    
    # Plotting KMeans
    plt.scatter(x=kmeans_data['PP%'], y=kmeans_data['PK%'], c=kmeans_data['kmeans_3'])
    plt.show()

    # Testing multiple clusters
    for k in range(1, 6):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(kmeans_data[['PP%', 'PK%']])
        kmeans_data[f'KMeans_{k}'] = kmeans.labels_
    
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20,5))

    for i, ax in enumerate(fig.axes, start=1):
        ax.scatter(x=kmeans_data['PP%'], y=kmeans_data['PK%'], c=kmeans_data[f'KMeans_{i}'])
        ax.set_title(f'N Clusters: {i}')
        ax.set_xlabel("Powerplay %")
        ax.set_ylabel("Penalty Kill %")
    
    plt.show()


goals_for_vs_goals_against()
wins_vs_losses()
powerplay_vs_penalty_kill()
powerplay_vs_wins()
age_vs_wins()
kmeans_goals_for_against()
