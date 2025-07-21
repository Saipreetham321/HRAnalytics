import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('employe.csv')
print("Initial Data Head")
print(df.head())
print("\nInitial Data Info")
print(df.info())


total_employees = df.shape[0]
employees_left = df[df['left'] == 1].shape[0]
print(f"\nWorkforce Size and Attrition Count")
print(f"Current workforce size: {total_employees}")
print(f"Number of employees who have left: {employees_left}")


attrition_rate = (employees_left / total_employees) * 100
print(f"\nOverall Attrition Rate")
print(f"Overall Attrition Rate: {attrition_rate:.2f}%")




print("\n Statistical Summary for Employees Who Stayed (left = 0) ")
print(df[df['left'] == 0].describe())


print("\n Statistical Summary for Employees Who Left (left = 1) ")
print(df[df['left'] == 1].describe())


print("\n Attrition by Department ")
dept_attrition = df.groupby('dept')['left'].value_counts(normalize=True).unstack().fillna(0)
dept_attrition['attrition_rate'] = dept_attrition[1] * 100
print(dept_attrition)


print("\n Attrition by Salary ")
salary_attrition = df.groupby('salary')['left'].value_counts(normalize=True).unstack().fillna(0)
salary_attrition['attrition_rate'] = salary_attrition[1] * 100
print(salary_attrition)


print("\n Attrition by Work Accident ")
accident_attrition = df.groupby('workAccident')['left'].value_counts(normalize=True).unstack().fillna(0)
accident_attrition['attrition_rate'] = accident_attrition[1] * 100
print(accident_attrition)


print("\n Attrition by Promotion in Last 5 Years ")
promotion_attrition = df.groupby('promotionInLast5years')['left'].value_counts(normalize=True).unstack().fillna(0)
promotion_attrition['attrition_rate'] = promotion_attrition[1] * 100
print(promotion_attrition)


sns.set_style("whitegrid")


plt.figure(figsize=(6, 4))
sns.countplot(x='left', data=df, palette='viridis')
plt.title('Overall Employee Attrition (0: Stayed, 1: Left)')
plt.xlabel('Attrition Status')
plt.ylabel('Number of Employees')
plt.xticks([0, 1], ['Stayed', 'Left'])
plt.show()


plt.figure(figsize=(12, 6))
sns.barplot(x=dept_attrition.index, y=dept_attrition['attrition_rate'], palette='coolwarm')
plt.title('Attrition Rate by Department')
plt.xlabel('Department')
plt.ylabel('Attrition Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
salary_order = ['low', 'medium', 'high']
sns.barplot(x=salary_attrition.index, y=salary_attrition['attrition_rate'], order=salary_order, palette='plasma')
plt.title('Attrition Rate by Salary Level')
plt.xlabel('Salary Level')
plt.ylabel('Attrition Rate (%)')
plt.tight_layout()
plt.show()

leavers_df = df[df['left'] == 1].copy() # Replace 'attrition_status' and 'Left' with your actual column name and value

pearson_corr = leavers_df['numberOfProjects'].corr(leavers_df['timeSpent.company'], method='pearson')
print(f"Pearson Correlation (Leavers): {pearson_corr}")

spearman_corr = leavers_df['numberOfProjects'].corr(leavers_df['timeSpent.company'], method='spearman')
print(f"Spearman Correlation (Leavers): {spearman_corr}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x='numberOfProjects', y='timeSpent.company', data=leavers_df)
plt.title('Number of Projects vs. Time Spent at Company for Employees Who Left')
plt.xlabel('Number of Projects')
plt.ylabel('Time Spent at Company (Years)')
plt.grid(True)
plt.show()
