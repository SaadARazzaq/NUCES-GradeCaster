import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from merger import merger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)

def evaluations_till_MID2(df):

    # Before Mid 2, We are only considering first 4 quizes and assignments

    As_dataFrame = df[['As:1', 'As:2', 'As:3', 'As:4']]
    Qz_dataFrame = df[['Qz:1', 'Qz:2', 'Qz:3', 'Qz:4']]
    S_I_dataFrame = df['S-I']

    As_Statistics = As_dataFrame.describe()
    Qz_Statistics = Qz_dataFrame.describe()
    S_I_Statistics = S_I_dataFrame.describe()

    As_Statistics.loc['iqr'] = As_Statistics.loc['75%'] - As_Statistics.loc['25%']
    As_Statistics.loc['std_dev'] = As_dataFrame.std()
    As_Statistics.loc['mean'] = As_dataFrame.mean()

    Qz_Statistics.loc['iqr'] = Qz_Statistics.loc['75%'] - Qz_Statistics.loc['25%']
    Qz_Statistics.loc['std_dev'] = Qz_dataFrame.std()
    Qz_Statistics.loc['mean'] = Qz_dataFrame.mean()

    S_I_Statistics['iqr'] = S_I_Statistics['75%'] - S_I_Statistics['25%']
    S_I_Statistics['std_dev'] = S_I_dataFrame.std()
    S_I_Statistics['mean'] = S_I_dataFrame.mean()

    return {
        'Assignments Stats': As_Statistics, 'Quizzes Stats': Qz_Statistics, 'Sessional 1 Stats': S_I_Statistics
    }

def evaluations_till_Finals(df):

    # For best of 5 assignments and quizes we are calculating overall
    
    As_dataFrame = df[['As']]
    Qz_dataFrame = df[['Qz']]
    S_I_dataFrame = df['S-I']
    S_II_dataFrame = df['S-II']

    As_Statistics = As_dataFrame.describe()
    Qz_Statistics = Qz_dataFrame.describe()
    S_I_Statistics = S_I_dataFrame.describe()
    S_II_Statistics = S_II_dataFrame.describe()
    
    As_Statistics.loc['iqr'] = As_Statistics.loc['75%'] - As_Statistics.loc['25%']
    As_Statistics.loc['std_dev'] = As_dataFrame.std()
    As_Statistics.loc['mean'] = As_dataFrame.mean()

    Qz_Statistics.loc['iqr'] = Qz_Statistics.loc['75%'] - Qz_Statistics.loc['25%']
    Qz_Statistics.loc['std_dev'] = Qz_dataFrame.std()
    Qz_Statistics.loc['mean'] = Qz_dataFrame.mean()

    S_I_Statistics['iqr'] = S_I_Statistics['75%'] - S_I_Statistics['25%']
    S_I_Statistics['std_dev'] = S_I_dataFrame.std()
    S_I_Statistics['mean'] = S_I_dataFrame.mean()

    S_II_Statistics['iqr'] = S_II_Statistics['75%'] - S_II_Statistics['25%']
    S_II_Statistics['std_dev'] = S_II_dataFrame.std()
    S_II_Statistics['mean'] = S_II_dataFrame.mean()

    return {
        'Assignments Stats': As_Statistics, 'Quizzes Stats': Qz_Statistics, 'Sessional 1 Stats': S_I_Statistics, 'Sessional 2 Stats': S_II_Statistics
    }

def EDA_before_MID2(df):

    # ------------------- BoxPlots

    st.subheader("Boxplot of First 4 Assignments")
    df[['As:1', 'As:2', 'As:3', 'As:4']].boxplot()
    plt.xlabel("Assignment Number")
    plt.ylabel("Marks")
    st.pyplot()

    st.subheader("Boxplot of First 4 Quizzes")
    df[['Qz:1', 'Qz:2', 'Qz:3', 'Qz:4']].boxplot()
    plt.xlabel("Quiz Number")
    plt.ylabel("Marks")
    st.pyplot()

    st.subheader("Boxplot of S-I Marks")
    df['S-I'].plot(kind='box')
    plt.xlabel("Sessional I")
    plt.ylabel("Marks")
    st.pyplot()

    # ------------------- Histograms

    st.subheader("Histogram of First 4 Assignments")
    df[['As:1', 'As:2', 'As:3', 'As:4']].hist(bins=10, figsize=(10, 6))
    plt.xlabel("Marks")
    plt.ylabel("Number of Students")
    st.pyplot()

    st.subheader("Histogram of First 4 Quizzes")
    df[['Qz:1', 'Qz:2', 'Qz:3', 'Qz:4']].hist(bins=10, figsize=(10, 6))
    plt.xlabel("Marks")
    plt.ylabel("Number of Students")
    st.pyplot()

    st.subheader("Histogram of S-I Marks")
    df['S-I'].hist(bins=10, figsize=(10, 6))
    plt.xlabel("Marks")
    plt.ylabel("Number of Students")
    st.pyplot()

    # ------------------- Density Plots

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    sns.kdeplot(data=df[['As:1', 'As:2', 'As:3', 'As:4']], ax=axs[0], fill=True)
    axs[0].set_title('Density Plot of First 4 Assignments')
    axs[0].set_xlabel("Marks")
    axs[0].set_ylabel("Density")

    sns.kdeplot(data=df[['Qz:1', 'Qz:2', 'Qz:3', 'Qz:4']], ax=axs[1], fill=True)
    axs[1].set_title('Density Plot of First 4 Quizzes')
    axs[1].set_xlabel("Marks")
    axs[1].set_ylabel("Density")

    sns.kdeplot(data=df['S-I'], ax=axs[2], fill=True)
    axs[2].set_title('Density Plot of S-I Marks')
    axs[2].set_xlabel("Marks")
    axs[2].set_ylabel("Density")

    plt.tight_layout()
    return fig

def EDA_before_Finals(df):
    
    # ------------------- BoxPlots

    st.subheader("Boxplot of Sum of Top 5 Assignments")
    df[['As']].boxplot()
    plt.xlabel("Total Students")
    plt.ylabel("Total Marks")
    st.pyplot()

    st.subheader("Boxplot of Sum of Top 5 Quizzes")
    df[['Qz']].boxplot()
    plt.xlabel("Total Students")
    plt.ylabel("Total Marks")
    st.pyplot()

    st.subheader("Boxplot of S-1 Marks")
    df['S-I'].plot(kind='box')
    plt.xlabel("Total Students")
    plt.ylabel("Total Marks")
    st.pyplot()

    st.subheader("Boxplot of S-2 Marks")
    df['S-II'].plot(kind='box')
    plt.xlabel("Total Students")
    plt.ylabel("Total Marks")
    st.pyplot()

    # ------------------- Histograms

    st.subheader("Histogram of Sum of Top 5 Assignments")
    df[['As']].hist(bins=10, figsize=(10, 6))
    plt.xlabel("Marks")
    plt.ylabel("Number of Students")
    st.pyplot()
    
    st.subheader("Histogram of Sum of Top 5 Quizzes")
    df[['Qz']].hist(bins=10, figsize=(10, 6))
    plt.xlabel("Marks")
    plt.ylabel("Number of Students")
    st.pyplot()

    st.subheader("Histogram of S-1 Marks")
    df['S-I'].hist(bins=10, figsize=(10, 6))
    plt.xlabel("Marks")
    plt.ylabel("Number of Students")
    st.pyplot()

    st.subheader("Histogram of S-2 Marks")
    df['S-II'].hist(bins=10, figsize=(10, 6))
    plt.xlabel("Marks")
    plt.ylabel("Number of Students")
    st.pyplot()

    # ------------------- Density Plots

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    sns.kdeplot(data=df[['As']], ax=axs[0, 0], fill=True)
    axs[0, 0].set_title('Density Plot of Sum of Top 5 Assignments')
    axs[0, 0].set_xlabel("Marks")
    axs[0, 0].set_ylabel("Density")

    sns.kdeplot(data=df[['Qz']], ax=axs[0, 1], fill=True)
    axs[0, 1].set_title('Density Plot of Sum of Top 5 Quizzes')
    axs[0, 1].set_xlabel("Marks")
    axs[0, 1].set_ylabel("Density")

    sns.kdeplot(data=df[['S-I']], ax=axs[1, 0], fill=True)
    axs[1, 0].set_title('Density Plot of S-1 Marks')
    axs[1, 0].set_xlabel("Marks")
    axs[1, 0].set_ylabel("Density")

    sns.kdeplot(data=df[['S-II']], ax=axs[1, 1], fill=True)
    axs[1, 1].set_title('Density Plot of S-2 Marks')
    axs[1, 1].set_xlabel("Marks")
    axs[1, 1].set_ylabel("Density")

    plt.tight_layout()
    return fig

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

# Models --------------------------------------------

def KNN_for_Mid2(df):
    
    df['All_As'] = df[['As:1', 'As:2', 'As:3', 'As:4']].sum(axis=1)
    df['All_Qz'] = df[['Qz:1', 'Qz:2', 'Qz:3', 'Qz:4']].sum(axis=1)

    X = df[['All_As', 'All_Qz', 'S-I']]
    y = df['Grade']

    le = LabelEncoder()  # Encode the target variable in the form of 0 and 1
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)  # Train the model
    
    y_pred = knn.predict(X_test)  # Make predictions on the testing set

    # Model Evaluation
    confusion_matrix_knn_mid2 = confusion_matrix(y_test, y_pred)
    classification_report_knn_mid2 = classification_report(y_test, y_pred, target_names=le.classes_)
    accuracy_knn_mid2 = accuracy_score(y_test, y_pred)

    return confusion_matrix_knn_mid2, classification_report_knn_mid2, accuracy_knn_mid2

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def KNN_for_Finals(df):

    X = df[['As', 'Qz', 'S-I', 'S-II']]
    y = df['Grade']

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    # Model Evaluation
    confusion_matrix_knn_finals = confusion_matrix(y_test, y_pred)
    classification_report_knn_finals = classification_report(y_test, y_pred, target_names=le.classes_)
    accuracy_knn_finals = accuracy_score(y_test, y_pred)

    return confusion_matrix_knn_finals, classification_report_knn_finals, accuracy_knn_finals

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def D_Tree_for_Mid2(df):
    
    df['All_As'] = df[['As:1', 'As:2', 'As:3', 'As:4']].sum(axis=1)
    df['All_Qz'] = df[['Qz:1', 'Qz:2', 'Qz:3', 'Qz:4']].sum(axis=1)

    # Split the data
    X = df[['All_As', 'All_Qz', 'S-I']]
    y = df['Grade']

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)  #  Train the Model

    y_pred = tree.predict(X_test)

    # Model Evaluation
    confusion_matrix_dtree_mid2 = confusion_matrix(y_test, y_pred)
    classification_report_dtree_mid2 = classification_report(y_test, y_pred, target_names=le.classes_)
    accuracy_dtree_mid2 = accuracy_score(y_test, y_pred)

    return confusion_matrix_dtree_mid2, classification_report_dtree_mid2, accuracy_dtree_mid2

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def D_Tree_for_Finals(df):

    X = df[['As', 'Qz', 'S-I', 'S-II']]
    y = df['Grade']

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)

    # Model Evaluation
    confusion_matrix_dtree_finals = confusion_matrix(y_test, y_pred)
    classification_report_dtree_finals = classification_report(y_test, y_pred, target_names=le.classes_)
    accuracy_dtree_finals = accuracy_score(y_test, y_pred)

    return confusion_matrix_dtree_finals, classification_report_dtree_finals, accuracy_dtree_finals

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

# Configure Streamlit page settings
st.set_page_config(
    page_title="NUCES GradeCrafter - Faculty",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  # Page layout option
)

def main():

 # Title in the middle
    st.title("NUCES GradeCrafter")
    st.subheader("Your Grading Companion")

    # Image at the upper side
    st.image("https://upload.wikimedia.org/wikipedia/en/e/e4/National_University_of_Computer_and_Emerging_Sciences_logo.png", use_column_width=False)

    st.sidebar.title("Navigation")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Read the uploaded file as a DataFrame
        df = pd.read_excel(uploaded_file)

        # Fit LabelEncoder
        le = LabelEncoder()
        le.fit(df['Grade'])

        # Sidebar navigation
        page = st.sidebar.selectbox("Go to", ["Evaluations till Mid-2", "Evaluations till Finals", "EDA till Mid-2", "EDA till Finals"])

        # Display selected page
        if page == "Evaluations till Mid-2":
            st.header("Evaluations till Mid-2")
            mid2_stats = evaluations_till_MID2(df)
            for key, value in mid2_stats.items():
                st.subheader(key)
                st.table(value)

            confusion_matrix_knn_mid2, classification_report_knn_mid2, accuracy_knn_mid2 = KNN_for_Mid2(df)
            st.subheader("Confusion Matrix (KNN - Mid-2)")
            st.table(pd.DataFrame(confusion_matrix_knn_mid2, index=le.classes_, columns=le.classes_))
            st.subheader("Classification Report (KNN - Mid-2)")
            st.markdown(f"```\n{classification_report_knn_mid2}\n```")
            st.subheader("Accuracy (KNN - Mid-2)")
            st.write(f"{accuracy_knn_mid2 * 100:.2f}%")

            confusion_matrix_dtree_mid2, classification_report_dtree_mid2, accuracy_dtree_mid2 = D_Tree_for_Mid2(df)
            st.subheader("Confusion Matrix (Decision Tree - Mid-2)")
            st.table(pd.DataFrame(confusion_matrix_dtree_mid2, index=le.classes_, columns=le.classes_))
            st.subheader("Classification Report (Decision Tree - Mid-2)")
            st.markdown(f"```\n{classification_report_dtree_mid2}\n```")
            st.subheader("Accuracy (Decision Tree - Mid-2)")
            st.write(f"{accuracy_dtree_mid2 * 100:.2f}%")

        elif page == "Evaluations till Finals":
            st.header("Evaluations till Finals")
            finals_stats = evaluations_till_Finals(df)
            for key, value in finals_stats.items():
                st.subheader(key)
                st.table(value)

            confusion_matrix_knn_finals, classification_report_knn_finals, accuracy_knn_finals = KNN_for_Finals(df)
            st.subheader("Confusion Matrix (KNN - Finals)")
            st.table(pd.DataFrame(confusion_matrix_knn_finals, index=le.classes_, columns=le.classes_))
            st.subheader("Classification Report (KNN - Finals)")
            st.markdown(f"```\n{classification_report_knn_finals}\n```")
            st.subheader("Accuracy (KNN - Finals)")
            st.write(f"{accuracy_knn_finals * 100:.2f}%")

            confusion_matrix_dtree_finals, classification_report_dtree_finals, accuracy_dtree_finals = D_Tree_for_Finals(df)
            st.subheader("Confusion Matrix (Decision Tree - Finals)")
            st.table(pd.DataFrame(confusion_matrix_dtree_finals, index=le.classes_, columns=le.classes_))
            st.subheader("Classification Report (Decision Tree - Finals)")
            st.markdown(f"```\n{classification_report_dtree_finals}\n```")
            st.subheader("Accuracy (Decision Tree - Finals)")
            st.write(f"{accuracy_dtree_finals * 100:.2f}%")

        
        elif page == "EDA till Mid-2":
            st.header("EDA till Mid-2")
            EDA_before_MID2(df)
            st.pyplot()

        elif page == "EDA till Finals":
            st.header("EDA till Finals")
            EDA_before_Finals(df)
            st.pyplot()

if __name__ == "__main__":
    main()