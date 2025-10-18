# Make sure you're in StudentPredictionApp
cd("StudentPerfomancePredictionApp")  # if not already there

# Create sample data file
using DataFrames, CSV, Random

function generate_sample_data()
    Random.seed!(42)
    n = 100  # 100 students
    
    df = DataFrame(
        student_id = 1:n,
        attendance_rate = clamp.(randn(n) .* 0.15 .+ 0.8, 0, 1),
        homework_completion = clamp.(randn(n) .* 0.2 .+ 0.75, 0, 1),
        test_score = clamp.(randn(n) .* 15 .+ 75, 0, 100),
        participation = rand(1:10, n),
        previous_gpa = clamp.(randn(n) .* 0.5 .+ 3.0, 0, 4.0)
    )
    
    # Create target: students are at-risk if multiple factors are low
    df.at_risk = (df.attendance_rate .< 0.7) .| 
                 (df.homework_completion .< 0.6) .| 
                 (df.test_score .< 65)
    
    filepath = joinpath("data", "samples", "student_data_sample.csv")
    CSV.write(filepath, df)
    println("✓ Sample data created: $filepath")
    return df
end

# Generate the sample data
sample_df = generate_sample_data()

# View first few rows
first(sample_df, 5)