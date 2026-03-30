"""
╔══════════════════════════════════════════════════════════════════╗
║          TITANIC SURVIVAL PREDICTION SYSTEM                      ║
║  Factors: Socio-Economic Status, Age, Gender & More              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  LOAD & EXPLORE
# ─────────────────────────────────────────────
df = pd.read_csv("Titanic-Dataset.csv")
print("\n" + "═"*60)
print("  TITANIC SURVIVAL PREDICTION SYSTEM")
print("═"*60)
print(f"\n  Dataset: {df.shape[0]} passengers, {df.shape[1]} features")
print(f"  Survived: {df['Survived'].sum()} ({df['Survived'].mean()*100:.1f}%)")
print(f"  Perished: {(1-df['Survived']).sum()} ({(1-df['Survived']).mean()*100:.1f}%)")

# ─────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(data):
    df = data.copy()

    # Title extraction from Name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    df['Title'] = df['Title'].map(title_map).fillna('Rare')

    # Age imputation by Title median
    df['Age'] = df.groupby('Title')['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Fare imputation
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Cabin: does the passenger have one?
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # Embarked
    df['Embarked'].fillna('S', inplace=True)

    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Age bins
    df['AgeBin'] = pd.cut(df['Age'],
        bins=[0, 12, 18, 35, 60, 100],
        labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])

    # Fare bins (proxy for wealth)
    df['FareBin'] = pd.qcut(df['Fare'], 4,
        labels=['Low', 'Medium', 'High', 'VeryHigh'])

    # Fare per person (group ticket normalisation)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # Gender × Class interaction
    df['SexPclass'] = df['Sex'] + '_' + df['Pclass'].astype(str)

    return df

df_eng = engineer_features(df)

# ─────────────────────────────────────────────
# 3.  ENCODE & PREPARE
# ─────────────────────────────────────────────
cat_cols = ['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin', 'SexPclass']
le = LabelEncoder()
for col in cat_cols:
    df_eng[col] = le.fit_transform(df_eng[col].astype(str))

features = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'Embarked', 'Title', 'HasCabin', 'FamilySize', 'IsAlone',
    'AgeBin', 'FareBin', 'FarePerPerson', 'SexPclass'
]

X = df_eng[features]
y = df_eng['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# 4.  TRAIN MODELS
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Random Forest':        RandomForestClassifier(n_estimators=300, max_depth=7,
                                min_samples_split=5, random_state=42),
    'Gradient Boosting':    GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                learning_rate=0.05, random_state=42),
    'Logistic Regression':  LogisticRegression(C=1.0, max_iter=1000, random_state=42)
}

print("\n" + "─"*60)
print("  MODEL PERFORMANCE  (5-fold cross-validation)")
print("─"*60)

scores = {}
for name, model in models.items():
    X_input = X_scaled if name == 'Logistic Regression' else X
    cv_scores = cross_val_score(model, X_input, y, cv=cv, scoring='accuracy')
    roc_scores = cross_val_score(model, X_input, y, cv=cv, scoring='roc_auc')
    scores[name] = {'acc': cv_scores.mean(), 'roc': roc_scores.mean()}
    print(f"  {name:<22}  Accuracy: {cv_scores.mean()*100:.1f}%  ±{cv_scores.std()*100:.1f}%  |  AUC: {roc_scores.mean():.3f}")

# Best model → Random Forest (fit on full data for feature importance)
rf = models['Random Forest']
rf.fit(X, y)

gb = models['Gradient Boosting']
gb.fit(X, y)

# ─────────────────────────────────────────────
# 5.  SURVIVAL PREDICTOR FUNCTION
# ─────────────────────────────────────────────
def predict_survival(pclass, sex, age, sibsp=0, parch=0, fare=None,
                     embarked='S', has_cabin=0):
    """
    Predict survival probability for a single passenger.

    Parameters
    ----------
    pclass   : 1, 2, or 3
    sex      : 'male' or 'female'
    age      : passenger age
    sibsp    : siblings/spouses aboard
    parch    : parents/children aboard
    fare     : ticket fare (None → median by class)
    embarked : 'S', 'C', or 'Q'
    has_cabin: 1 if cabin known, else 0
    """
    if fare is None:
        fare = df.groupby('Pclass')['Fare'].median()[pclass]

    # Derive engineered features
    family_size   = sibsp + parch + 1
    is_alone      = int(family_size == 1)
    fare_per_person = fare / family_size

    # Title heuristic
    if sex == 'female':
        title = 'Mrs' if age >= 18 else 'Miss'
    else:
        title = 'Master' if age < 15 else 'Mr'

    age_bin = ('Child' if age <= 12 else 'Teen' if age <= 18 else
               'YoungAdult' if age <= 35 else 'Adult' if age <= 60 else 'Senior')
    fare_quartiles = df['Fare'].quantile([0.25, 0.50, 0.75]).values
    fare_bin = ('Low' if fare <= fare_quartiles[0] else
                'Medium' if fare <= fare_quartiles[1] else
                'High' if fare <= fare_quartiles[2] else 'VeryHigh')
    sex_pclass = f"{sex}_{pclass}"

    row = {
        'Pclass': pclass, 'Sex': sex, 'Age': age, 'SibSp': sibsp,
        'Parch': parch, 'Fare': fare, 'Embarked': embarked,
        'Title': title, 'HasCabin': has_cabin,
        'FamilySize': family_size, 'IsAlone': is_alone,
        'AgeBin': age_bin, 'FareBin': fare_bin,
        'FarePerPerson': fare_per_person, 'SexPclass': sex_pclass
    }

    tmp = pd.DataFrame([row])
    for col in cat_cols:
        tmp[col] = le.fit_transform(tmp[col].astype(str))

    # Ensure column order
    tmp = tmp[features]

    rf_prob  = rf.predict_proba(tmp)[0][1]
    gb_prob  = gb.predict_proba(tmp)[0][1]
    avg_prob = (rf_prob + gb_prob) / 2

    verdict = "✅ LIKELY SURVIVED" if avg_prob >= 0.5 else "❌ LIKELY PERISHED"
    return avg_prob, verdict

# ─────────────────────────────────────────────
# 6.  DEMO PREDICTIONS
# ─────────────────────────────────────────────
print("\n" + "─"*60)
print("  SAMPLE SURVIVAL PREDICTIONS")
print("─"*60)

test_cases = [
    dict(pclass=1, sex='female', age=28, fare=100, embarked='C', has_cabin=1,
         label="1st-class woman, age 28, Cherbourg"),
    dict(pclass=3, sex='male',   age=22, fare=7.25, embarked='S',
         label="3rd-class man, age 22, Southampton"),
    dict(pclass=1, sex='male',   age=45, sibsp=1, parch=0, fare=83, has_cabin=1,
         label="1st-class man, age 45, married"),
    dict(pclass=2, sex='female', age=8,  parch=1, fare=29,
         label="2nd-class girl, age 8, with parent"),
    dict(pclass=3, sex='female', age=30, sibsp=2, parch=3, fare=15.5,
         label="3rd-class woman, age 30, large family"),
    dict(pclass=1, sex='male',   age=60, fare=263, has_cabin=1,
         label="1st-class elderly man, premium fare"),
]

for tc in test_cases:
    label = tc.pop('label')
    prob, verdict = predict_survival(**tc)
    bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
    print(f"\n  {label}")
    print(f"  [{bar}] {prob*100:.1f}%  {verdict}")

# ─────────────────────────────────────────────
# 7.  FACTOR ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "─"*60)
print("  KEY SURVIVAL FACTORS")
print("─"*60)

# Survival rates by category
factors = {
    "Gender":     df.groupby('Sex')['Survived'].mean(),
    "Pclass":     df.groupby('Pclass')['Survived'].mean(),
    "Embarked":   df.groupby('Embarked')['Survived'].mean(),
}

for factor, rates in factors.items():
    print(f"\n  {factor}:")
    for cat, rate in rates.items():
        bar = "█" * int(rate * 30)
        print(f"    {str(cat):<12} {bar:<30} {rate*100:.1f}%")

# Age group
df['AgeBin_raw'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100],
                           labels=['Child','Teen','YoungAdult','Adult','Senior'])
age_surv = df.groupby('AgeBin_raw')['Survived'].mean().dropna()
print(f"\n  Age Groups:")
for grp, rate in age_surv.items():
    bar = "█" * int(rate * 30)
    print(f"    {str(grp):<12} {bar:<30} {rate*100:.1f}%")

# ─────────────────────────────────────────────
# 8.  VISUALIZATIONS
# ─────────────────────────────────────────────
# Color palette
C_SURV  = '#2ecc71'   # green   – survived
C_DIED  = '#e74c3c'   # red     – perished
C_ACC1  = '#3498db'   # blue
C_ACC2  = '#9b59b6'   # purple
C_BG    = '#f8f9fa'
C_DARK  = '#2c3e50'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor': C_BG,
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#cccccc',
    'axes.labelcolor': C_DARK,
    'xtick.color': C_DARK,
    'ytick.color': C_DARK,
    'text.color': C_DARK,
})

# ── Figure 1: Factor Analysis Dashboard ──────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')
gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

fig.text(0.5, 0.97, "Titanic Survival Analysis  —  Key Factors",
         ha='center', va='top', fontsize=18, fontweight='bold', color=C_DARK)
fig.text(0.5, 0.945, f"891 passengers  |  342 survived (38.4%)  |  549 perished (61.6%)",
         ha='center', va='top', fontsize=11, color='#7f8c8d')

# ── 1. Overall survival pie ───────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
sizes  = [df['Survived'].sum(), (1 - df['Survived']).sum()]
labels = ['Survived\n342 (38.4%)', 'Perished\n549 (61.6%)']
colors = [C_SURV, C_DIED]
wedges, texts = ax0.pie(sizes, labels=labels, colors=colors,
                         startangle=90, wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2))
for t in texts:
    t.set_fontsize(10)
ax0.set_title('Overall Outcome', fontsize=12, fontweight='bold', pad=10)

# ── 2. Gender survival ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
gender_data = df.groupby('Sex')['Survived'].value_counts(normalize=True).unstack()
gender_data = gender_data[[0, 1]].rename(columns={0: 'Perished', 1: 'Survived'})
gender_data[['Perished', 'Survived']].plot(kind='bar', ax=ax1,
    color=[C_DIED, C_SURV], edgecolor='white', linewidth=1.5, rot=0, width=0.6)
ax1.set_title('Survival by Gender', fontsize=12, fontweight='bold')
ax1.set_ylabel('Proportion', fontsize=10)
ax1.set_xlabel('')
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.legend(frameon=False, fontsize=9)
for p in ax1.patches:
    if p.get_height() > 0.05:
        ax1.text(p.get_x() + p.get_width()/2, p.get_height() + 0.01,
                 f'{p.get_height()*100:.0f}%', ha='center', fontsize=9, fontweight='bold')

# ── 3. Pclass survival ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
class_data = df.groupby('Pclass')['Survived'].value_counts(normalize=True).unstack()
class_data = class_data[[0, 1]].rename(columns={0: 'Perished', 1: 'Survived'})
class_labels = ['1st Class\n(Upper)', '2nd Class\n(Middle)', '3rd Class\n(Lower)']
class_data.index = class_labels
class_data[['Perished', 'Survived']].plot(kind='bar', ax=ax2,
    color=[C_DIED, C_SURV], edgecolor='white', linewidth=1.5, rot=0, width=0.6)
ax2.set_title('Survival by Socio-Economic Class', fontsize=12, fontweight='bold')
ax2.set_ylabel('Proportion', fontsize=10)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.legend(frameon=False, fontsize=9)
for p in ax2.patches:
    if p.get_height() > 0.05:
        ax2.text(p.get_x() + p.get_width()/2, p.get_height() + 0.01,
                 f'{p.get_height()*100:.0f}%', ha='center', fontsize=9, fontweight='bold')

# ── 4. Age distribution by survival ──────────────────────────────
ax3 = fig.add_subplot(gs[1, 0:2])
bins  = np.arange(0, 85, 5)
surv  = df[df['Survived'] == 1]['Age'].dropna()
died  = df[df['Survived'] == 0]['Age'].dropna()
ax3.hist(died, bins=bins, color=C_DIED, alpha=0.7, label='Perished', edgecolor='white')
ax3.hist(surv, bins=bins, color=C_SURV, alpha=0.7, label='Survived', edgecolor='white')
ax3.axvline(surv.median(), color=C_SURV, lw=2, linestyle='--', label=f'Survived median ({surv.median():.0f}y)')
ax3.axvline(died.median(), color=C_DIED, lw=2, linestyle='--', label=f'Perished median ({died.median():.0f}y)')
ax3.set_title('Age Distribution by Survival Outcome', fontsize=12, fontweight='bold')
ax3.set_xlabel('Age (years)', fontsize=10)
ax3.set_ylabel('Count', fontsize=10)
ax3.legend(frameon=False, fontsize=9)

# ── 5. Feature importance ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
feature_labels = {
    'SexPclass': 'Gender × Class', 'Sex': 'Gender', 'Pclass': 'Class',
    'Title': 'Title (Mr/Mrs…)', 'FareBin': 'Fare Tier',
    'Fare': 'Ticket Fare', 'FarePerPerson': 'Fare/Person',
    'Age': 'Age', 'AgeBin': 'Age Group', 'HasCabin': 'Has Cabin',
    'Embarked': 'Port of Embark', 'FamilySize': 'Family Size',
    'IsAlone': 'Travelling Alone', 'SibSp': 'Siblings/Spouse',
    'Parch': 'Parents/Children',
}
imp.index = [feature_labels.get(i, i) for i in imp.index]
colors_fi = [C_ACC1 if v > imp.median() else '#aab7c4' for v in imp.values]
bars = ax4.barh(imp.index, imp.values, color=colors_fi, edgecolor='white', height=0.7)
ax4.set_title('Feature Importance\n(Random Forest)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Importance', fontsize=10)
for bar, val in zip(bars, imp.values):
    ax4.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=7.5)
ax4.tick_params(axis='y', labelsize=8)

# ── 6. Heat map: Gender × Class ──────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
hmap = df.groupby(['Sex', 'Pclass'])['Survived'].mean().unstack()
im = ax5.imshow(hmap.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax5.set_xticks([0,1,2]); ax5.set_xticklabels(['1st', '2nd', '3rd'])
ax5.set_yticks([0,1]);   ax5.set_yticklabels(['Female', 'Male'])
ax5.set_title('Survival Rate:\nGender × Class', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04, format='%.0%%')
for i in range(2):
    for j in range(3):
        ax5.text(j, i, f'{hmap.values[i,j]*100:.0f}%',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 color='white' if hmap.values[i,j] < 0.4 or hmap.values[i,j] > 0.75 else C_DARK)

# ── 7. Family size survival ───────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
fam_surv = df.copy()
fam_surv['FamilySize'] = fam_surv['SibSp'] + fam_surv['Parch'] + 1
fam_rate  = fam_surv.groupby('FamilySize')['Survived'].mean()
fam_count = fam_surv.groupby('FamilySize')['Survived'].count()
bars6 = ax6.bar(fam_rate.index, fam_rate.values,
                color=[C_SURV if r > 0.5 else C_DIED for r in fam_rate.values],
                edgecolor='white', linewidth=1.5, width=0.6)
ax6.axhline(0.5, color='#7f8c8d', linestyle='--', linewidth=1.2, alpha=0.8)
ax6.set_title('Survival Rate by Family Size', fontsize=12, fontweight='bold')
ax6.set_xlabel('Family Size (self + relatives)', fontsize=10)
ax6.set_ylabel('Survival Rate', fontsize=10)
ax6.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax6.set_xticks(fam_rate.index)
for bar, cnt in zip(bars6, fam_count.values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'n={cnt}', ha='center', fontsize=7.5, color='#555')

# ── 8. Model comparison ───────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
model_names  = list(scores.keys())
accs  = [scores[m]['acc'] for m in model_names]
aucs  = [scores[m]['roc'] for m in model_names]
x     = np.arange(len(model_names))
w     = 0.35
b1 = ax7.bar(x - w/2, accs, w, label='Accuracy', color=C_ACC1, edgecolor='white')
b2 = ax7.bar(x + w/2, aucs, w, label='ROC-AUC',  color=C_ACC2, edgecolor='white')
ax7.set_title('Model Comparison\n(5-fold CV)', fontsize=12, fontweight='bold')
ax7.set_xticks(x)
short_names = ['Rnd Forest', 'Grad Boost', 'Log Reg']
ax7.set_xticklabels(short_names, fontsize=9)
ax7.set_ylim(0.7, 1.0)
ax7.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax7.legend(frameon=False, fontsize=9)
for bar in list(b1) + list(b2):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{bar.get_height()*100:.1f}%', ha='center', fontsize=8, fontweight='bold')

plt.savefig('titanic_factor_analysis.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("\n  ✅ Saved: titanic_factor_analysis.png")

# ── Figure 2: Survival Prediction Profiles ───────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.patch.set_facecolor('white')
fig2.suptitle('Survival Probability Profiles', fontsize=16, fontweight='bold', y=1.02)

# (a) Age × Gender × Class
ax = axes[0]
ages = np.arange(1, 80)
for sex, ls in [('female', '-'), ('male', '--')]:
    for pclass, col in [(1, '#1a5276'), (2, '#2980b9'), (3, '#aed6f1')]:
        probs = [predict_survival(pclass, sex, a, fare=df.groupby('Pclass')['Fare'].median()[pclass])[0]
                 for a in ages]
        label = f"{'F' if sex=='female' else 'M'} – {pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'}"
        ax.plot(ages, probs, lw=2, linestyle=ls, color=col, label=label)
ax.axhline(0.5, color='grey', lw=1, linestyle=':', alpha=0.7)
ax.set_title('Probability by Age\n(Gender & Class)', fontsize=11, fontweight='bold')
ax.set_xlabel('Age (years)'); ax.set_ylabel('Survival Probability')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(fontsize=8, frameon=False, ncol=2)
ax.set_facecolor(C_BG)

# (b) Fare × Class
ax = axes[1]
fares = np.linspace(5, 300, 100)
for pclass, col in [(1, '#1e8449'), (2, '#27ae60'), (3, '#a9dfbf')]:
    for sex, ls in [('female', '-'), ('male', '--')]:
        probs = [predict_survival(pclass, sex, 30, fare=f)[0] for f in fares]
        label = f"{'F' if sex=='female' else 'M'} – Class {pclass}"
        ax.plot(fares, probs, lw=2, linestyle=ls, color=col, label=label)
ax.axhline(0.5, color='grey', lw=1, linestyle=':', alpha=0.7)
ax.set_title('Probability by Fare\n(Age=30)', fontsize=11, fontweight='bold')
ax.set_xlabel('Ticket Fare (£)'); ax.set_ylabel('Survival Probability')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(fontsize=8, frameon=False, ncol=2)
ax.set_facecolor(C_BG)

# (c) Family size × Gender
ax = axes[2]
fam_sizes = range(1, 9)
for sex, col in [('female', C_SURV), ('male', C_DIED)]:
    for pclass, ls in [(1, '-'), (2, '--'), (3, ':')]:
        probs = []
        for fs in fam_sizes:
            sibsp = max(0, fs - 1)
            p, _ = predict_survival(pclass, sex, 30, sibsp=sibsp)
            probs.append(p)
        ax.plot(list(fam_sizes), probs, lw=2, ls=ls, color=col,
                label=f"{'F' if sex=='female' else 'M'} – Class {pclass}")
ax.axhline(0.5, color='grey', lw=1, linestyle=':', alpha=0.7)
ax.set_title('Probability by Family Size\n(Age=30)', fontsize=11, fontweight='bold')
ax.set_xlabel('Family Size'); ax.set_ylabel('Survival Probability')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(fontsize=8, frameon=False, ncol=2)
ax.set_xticks(list(fam_sizes))
ax.set_facecolor(C_BG)

plt.tight_layout()
plt.savefig('titanic_survival_profiles.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("  ✅ Saved: titanic_survival_profiles.png")

# ─────────────────────────────────────────────
# 9.  FINAL INSIGHTS SUMMARY
# ─────────────────────────────────────────────
print("\n" + "═"*60)
print("  KEY INSIGHTS — WHAT DROVE SURVIVAL?")
print("═"*60)

insights = [
    ("🚺 Gender",          "Women survived at 74% vs men at 19%. 'Women & children first' was real."),
    ("💎 Socio-Econ Class","1st class: 63% | 2nd class: 47% | 3rd class: 24%. Wealth = access to boats."),
    ("👦 Children (≤12)",  "Children had the highest survival among all age groups (~58%)."),
    ("🎫 Fare / Cabin",    "Higher fare & having a cabin strongly correlated with survival."),
    ("👨‍👩‍👧 Family Size",  "Solo travelers and very large families (7+) fared worst."),
    ("⚓ Port of Embark",  "Cherbourg passengers survived at 55% vs Southampton at 34%."),
    ("🏅 Title",           "Title (Mr vs Master vs Mrs) was one of the top predictors."),
    ("🔗 Interaction",     "Gender × Class combo was the single most predictive feature."),
]

for icon_label, detail in insights:
    print(f"\n  {icon_label}")
    print(f"  → {detail}")

print("\n" + "═"*60)
print("  SYSTEM READY — call predict_survival() for any passenger")
print("═"*60 + "\n")