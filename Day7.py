import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

data = {
    'movie_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'title': [
        'The Matrix', 'John Wick', 'The Godfather', 'Pulp Fiction', 'The Dark Knight',
        'Inception', 'Interstellar', 'The Avengers', 'Forrest Gump', 'Spirited Away',
        'Parasite', 'Fight Club', 'Blade Runner 2049', 'Mad Max: Fury Road', 'Whiplash',
        'La La Land', 'The Grand Budapest Hotel', 'Django Unchained', 'Inglourious Basterds', 'Once Upon a Time in Hollywood'
    ],
    'genre': [
        'Action, Sci-Fi', 'Action, Thriller', 'Crime, Drama', 'Crime, Drama, Thriller', 'Action, Crime, Drama',
        'Action, Sci-Fi, Thriller', 'Adventure, Drama, Sci-Fi', 'Action, Adventure, Sci-Fi',
        'Comedy, Drama, Romance', 'Animation, Adventure, Family, Fantasy',
        'Comedy, Drama, Thriller', 'Drama, Thriller', 'Sci-Fi, Thriller, Drama', 'Action, Adventure, Sci-Fi', 'Drama, Music',
        'Comedy, Drama, Music, Romance', 'Adventure, Comedy, Drama', 'Drama, Western', 'Adventure, Drama, War', 'Comedy, Drama'
    ],
    'director': [
        'Lana Wachowski, Lilly Wachowski', 'Chad Stahelski', 'Francis Ford Coppola', 'Quentin Tarantino', 'Christopher Nolan',
        'Christopher Nolan', 'Christopher Nolan', 'Joss Whedon', 'Robert Zemeckis', 'Hayao Miyazaki',
        'Bong Joon Ho', 'David Fincher', 'Denis Villeneuve', 'George Miller', 'Damien Chazelle',
        'Damien Chazelle', 'Wes Anderson', 'Quentin Tarantino', 'Quentin Tarantino', 'Quentin Tarantino'
    ],
    'actors': [
        'Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving',
        'Keanu Reeves, Michael Nyqvist, Alfie Allen, Willem Dafoe',
        'Marlon Brando, Al Pacino, James Caan, Diane Keaton',
        'John Travolta, Uma Thurman, Samuel L. Jackson, Bruce Willis',
        'Christian Bale, Heath Ledger, Aaron Eckhart, Michael Caine',
        'Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page, Tom Hardy',
        'Matthew McConaughey, Anne Hathaway, Jessica Chastain, Michael Caine',
        'Robert Downey Jr., Chris Evans, Scarlett Johansson, Mark Ruffalo',
        'Tom Hanks, Robin Wright, Gary Sinise, Sally Field',
        'Rumi Hiiragi, Miyu Irino, Mari Natsuki, Takeshi Naito',
        'Song Kang-ho, Lee Sun-kyun, Cho Yeo-jeong, Choi Woo-shik',
        'Brad Pitt, Edward Norton, Helena Bonham Carter, Meat Loaf',
        'Ryan Gosling, Harrison Ford, Ana de Armas, Sylvia Hoeks',
        'Tom Hardy, Charlize Theron, Nicholas Hoult, Hugh Keays-Byrne',
        'Miles Teller, J.K. Simmons, Paul Reiser, Melissa Benoist',
        'Ryan Gosling, Emma Stone, Rosemarie DeWitt, J.K. Simmons',
        'Ralph Fiennes, F. Murray Abraham, Mathieu Amalric, Adrien Brody',
        'Jamie Foxx, Christoph Waltz, Leonardo DiCaprio, Kerry Washington',
        'Brad Pitt, Mélanie Laurent, Christoph Waltz, Eli Roth',
        'Leonardo DiCaprio, Brad Pitt, Margot Robbie, Emile Hirsch'
    ],
    'keywords': [
        'virtual reality, AI, prophecy, chosen one, dystopia, simulation',
        'assassin, revenge, dog, hitman, action packed, underworld',
        'mafia, family, crime syndicate, power, betrayal, italian american',
        'non-linear, hitmen, diner, gangsters, dark comedy, pop culture',
        'batman, joker, chaos, vigilante, moral dilemma, gotham city',
        'dreams, subconscious, heist, mind-bending, layered reality, corporate espionage',
        'space travel, black hole, time dilation, survival, humanity, wormhole',
        'superheroes, team, alien invasion, earth protection, marvel, avengers initiative',
        'life story, historical events, love, destiny, american history, shrimp',
        'spirits, bathhouse, fantasy world, courage, coming of age, japanese folklore',
        'social inequality, class struggle, dark comedy, con artists, family, basement',
        'insomnia, consumerism, identity crisis, unreliable narrator, anarchy, soap',
        'replicant, dystopian future, AI, memory, identity, neo-noir',
        'post-apocalyptic, desert, chase, tyrant, survival, high-octane',
        'jazz, drumming, ambition, mentor, obsession, music school',
        'los angeles, aspiring artists, dreams, hollywood, musical, romance',
        'eccentric, hotel, concierge, caper, europe, nostalgia',
        'slavery, bounty hunter, revenge, antebellum south, spaghetti western',
        'world war ii, nazi hunting, revenge, alternate history, cinema',
        '1960s hollywood, fading star, stunt double, manson family, film industry'
    ],
    'description': [
        'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.',
        'An ex-hitman comes out of retirement to track down the gangsters that took everything from him.',
        'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
        'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
        'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
        'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
        'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
        'Earth\'s mightiest heroes must come together and learn to fight as a team if they are going to stop the mischievous Loki and his alien army from enslaving humanity.',
        'The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75, whose only desire is to be reunited with his childhood sweetheart.',
        'During her family\'s move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits, and where humans are changed into beasts.',
        'Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.',
        'An insomniac office worker looking for a way to change his life crosses paths with a devil-may-care soap maker and they form an underground fight club that evolves into something much, much more.',
        'Young Blade Runner K\'s discovery of a long-buried secret leads him to track down former Blade Runner Rick Deckard, who\'s been missing for thirty years.',
        'In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland with the help of a group of female prisoners, a psychotic worshiper, and a drifter named Max.',
        'A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student\'s potential.',
        'While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future.',
        'The adventures of Gustave H, a legendary concierge at a famous hotel from the fictional Republic of Zubrowka between the first and second World Wars, and Zero Moustafa, the lobby boy who becomes his most trusted friend.',
        'With the help of a German bounty-hunter, a freed slave sets out to rescue his wife from a brutal Mississippi plantation owner.',
        'In Nazi-occupied France during World War II, a plan to assassinate Nazi leaders by a group of Jewish U.S. soldiers coincides with a theatre owner\'s vengeful plans for the same.',
        'A faded television actor and his stunt double strive to achieve fame and success in the final years of Hollywood\'s Golden Age in 1969 Los Angeles.'
    ],
    'release_year': [
        1999, 2014, 1972, 1994, 2008, 2010, 2014, 2012, 1994, 2001, 2019, 1999,
        2017, 2015, 2014, 2016, 2014, 2012, 2009, 2019
    ]
}

df = pd.DataFrame(data)

def process_text_list(text_input):
    text = str(text_input).lower()
    items = text.split(',')
    processed_items = []
    for item in items:
        item = item.strip()
        item = re.sub(r'[^a-z0-9]', '', item)
        if item:
            processed_items.append(item)
    return " ".join(processed_items)

def clean_text_block(text_input):
    text = str(text_input).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['processed_genre'] = df['genre'].apply(process_text_list)
df['processed_director'] = df['director'].apply(process_text_list)
df['processed_actors'] = df['actors'].apply(process_text_list)
df['processed_keywords'] = df['keywords'].apply(process_text_list)
df['processed_description'] = df['description'].apply(clean_text_block)

df['soup'] = (
    df['processed_genre'] + ' ' +
    df['processed_director'] + ' ' +
    df['processed_actors'] + ' ' +
    df['processed_keywords'] + ' ' +
    df['processed_description']
)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.8)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['soup'])
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

titles_to_indices_map = pd.Series(df.index, index=df['title'].str.lower())

def get_recommendations(input_title, num_recommendations=5, data_frame=df, similarity_matrix=cosine_sim_matrix, indices_map=titles_to_indices_map):
    input_title_lower = input_title.lower()
    if input_title_lower not in indices_map:
        print(f"Movie '{input_title}' not found in the dataset.")
        available_titles = ", ".join(sorted(list(data_frame['title'].unique()))[:10]) + "..."
        print(f"Available movie titles include: {available_titles}")
        return pd.DataFrame()

    idx = indices_map[input_title_lower]
    
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    top_scores = sim_scores[1:num_recommendations + 1]
    
    if not top_scores:
        print(f"No other sufficiently similar movies found to recommend for '{input_title}'.")
        return pd.DataFrame()

    movie_indices = [i[0] for i in top_scores]
    movie_similarity_values = [round(i[1], 4) for i in top_scores]
    
    recommended_movies_df = data_frame.iloc[movie_indices][['title', 'genre', 'director', 'actors', 'release_year']].copy()
    recommended_movies_df['similarity_score'] = movie_similarity_values
    
    return recommended_movies_df.reset_index(drop=True)

print("Enriched Movie Data Sample (First 3):")
print(df[['title', 'genre', 'director', 'actors', 'keywords', 'release_year', 'description']].head(3))
print("\nProcessed Feature Soup Sample (First movie):")
print(f"'{df['title'].iloc[0]}': {df['soup'].iloc[0][:300]}...")


test_cases = [
    {'title': 'The Matrix', 'num': 3},
    {'title': 'Inception', 'num': 4},
    {'title': 'Pulp Fiction', 'num': 3},
    {'title': 'John Wick', 'num': 3},
    {'title': 'La La Land', 'num': 3},
    {'title': 'Spirited Away', 'num': 3},
    {'title': 'NonExistent Movie', 'num': 3}
]

for case in test_cases:
    title = case['title']
    num = case['num']
    print(f"\nRecommendations for '{title}' (Top {num}):")
    recommendations = get_recommendations(title, num_recommendations=num)
    if not recommendations.empty:
        print(recommendations)