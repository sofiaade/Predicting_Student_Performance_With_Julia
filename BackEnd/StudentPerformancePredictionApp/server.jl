using Genie, Genie.Router, Genie.Renderer.Json
using Genie.Requests
using CSV, DataFrames, Dates
using JSON3
using UUIDs
using Logging

# Enable CORS
Genie.config.cors_headers["Access-Control-Allow-Origin"] = "*"
Genie.config.cors_headers["Access-Control-Allow-Headers"] = "Content-Type"
Genie.config.cors_headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
Genie.config.cors_allowed_origins = ["*"]

# Configuration
const CONFIG = Dict(
    "max_file_size" => 10_000_000,  # 10MB
    "allowed_extensions" => [".csv"],
    "upload_dir" => "data/uploads",
    "python_script" => joinpath(@__DIR__, "ml_models.py"),
    "session_timeout" => 3600  # 1 hour
)

# Session-based storage (replaces globals)
const DataStore = Dict{String, Dict{String, Any}}()
const SessionTimestamps = Dict{String, DateTime}()

# Cleanup old sessions
function cleanup_old_sessions()
    current_time = now()
    expired = String[]
    
    for (session_id, timestamp) in SessionTimestamps
        if (current_time - timestamp).value / 1000 > CONFIG["session_timeout"]
            push!(expired, session_id)
        end
    end
    
    for session_id in expired
        delete!(DataStore, session_id)
        delete!(SessionTimestamps, session_id)
        @info "Cleaned up expired session" session_id
    end
end

# Ensure directories exist
function ensure_directories()
    if !isdir(CONFIG["upload_dir"])
        mkpath(CONFIG["upload_dir"])
        @info "Created upload directory" path=CONFIG["upload_dir"]
    end
end

# Validate file upload
function validate_file(file)
    # Check file extension
    filename = file.name
    ext = lowercase(splitext(filename)[2])
    
    if !(ext in CONFIG["allowed_extensions"])
        return (false, "Only CSV files are allowed")
    end
    
    # Check file size
    if length(file.data) > CONFIG["max_file_size"]
        return (false, "File too large. Maximum size is 10MB")
    end
    
    # Check if file is empty
    if length(file.data) == 0
        return (false, "File is empty")
    end
    
    return (true, "")
end

# CORS preflight handlers
route("/api/upload", method = OPTIONS) do
    ""
end

route("/api/train", method = OPTIONS) do
    ""
end

route("/api/results", method = OPTIONS) do
    ""
end

# Upload endpoint - Julia handles data preprocessing
route("/api/upload", method = POST) do
    try
        # Cleanup old sessions periodically
        cleanup_old_sessions()
        
        if !haskey(filespayload(), "datafile")
            return json(Dict("status" => "error", "message" => "No file uploaded"))
        end
        
        file = filespayload("datafile")
        
        # Validate file
        is_valid, error_msg = validate_file(file)
        if !is_valid
            @warn "File validation failed" error=error_msg filename=file.name
            return json(Dict("status" => "error", "message" => error_msg))
        end
        # Generate session and timestamp
        session_id = string(uuid4())
        timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")

       # Save original file - THIS IS WHAT PYTHON WILL USE
       filepath = joinpath(CONFIG["upload_dir"], "upload_$(timestamp)_$(session_id).csv")
       write(filepath, file.data)
       @info "✓ File saved" filename=file.name size=length(file.data) session=session_id
       
       # Read ONLY to verify and get info - DON'T SAVE IT BACK
       try
           data = CSV.read(filepath, DataFrame)
           @info "✓ Column names in uploaded file: $(names(data))"
           row_count = nrow(data)
       catch e
           @error "Failed to read CSV" exception=e
           return json(Dict("status" => "error", "message" => "Invalid CSV format"))
       end
       
       # Use original file path directly - NO processed file at all
       processed_path = filepath
      
        # Store in session
        DataStore[session_id] = Dict(
            "data" => data,
            "filepath" => processed_path,
            "original_filename" => file.name,
            "timestamp" => now()
        )
        SessionTimestamps[session_id] = now()
        
        return json(Dict(
            "status" => "success",
            "message" => "Data uploaded successfully",
            "session_id" => session_id,
            "rows" => nrow(data),
            "columns" => names(data)
        ))
        
    catch e
        @error "Upload error" exception=(e, catch_backtrace())
        
        # Provide more specific error messages
        error_msg = if isa(e, CSV.Error)
            "Invalid CSV format: $(e.msg)"
        elseif isa(e, ArgumentError)
            "Data validation error: $(e.msg)"
        else
            "Upload failed: $(string(e))"
        end
        
        return json(Dict("status" => "error", "message" => error_msg))
    end
end

# Train endpoint - Calls Python for ML
route("/api/train", method = POST) do
    try
        payload = jsonpayload()
        
        # Get session ID
        if !haskey(payload, "session_id")
            return json(Dict("status" => "error", "message" => "No session_id provided"))
        end
        
        session_id = payload["session_id"]
        
        # Check if session exists
        if !haskey(DataStore, session_id)
            return json(Dict(
                "status" => "error", 
                "message" => "Session expired or invalid. Please upload data again."
            ))
        end
        
        # Get model type
        model_type = get(payload, "model_type", "logistic")
        
        # Validate model type
        valid_models = ["logistic", "decision_tree", "linear"]
        if !(model_type in valid_models)
            return json(Dict(
                "status" => "error",
                "message" => "Invalid model type. Must be one of: $(join(valid_models, ", "))"
            ))
        end
        
        filepath = DataStore[session_id]["filepath"]
        
        @info "✓ Starting ML training" session=session_id model=model_type
        
        # Call Python script with proper error handling
        python_script = CONFIG["python_script"]
        
        if !isfile(python_script)
            @error "Python script not found" path=python_script
            return json(Dict(
                "status" => "error",
                "message" => "ML script not found. Please check server configuration."
            ))
        end
        
        # Run Python and capture output
        cmd = `python3 $python_script $filepath $model_type`
        
        output = try
            read(cmd, String)
        catch e
            @error "Python execution failed" exception=(e, catch_backtrace())
            return json(Dict(
                "status" => "error",
                "message" => "ML training failed. Please check your data format."
            ))
        end
        
        @info "✓ Python processing complete"
        
        # Parse JSON result from Python
        results = try
            JSON3.read(output)
        catch e
            @error "Failed to parse Python output" output=output exception=e
            return json(Dict(
                "status" => "error",
                "message" => "Failed to parse ML results"
            ))
        end
        
        if results["status"] == "success"
            # Store results in session
            DataStore[session_id]["results"] = results
            SessionTimestamps[session_id] = now()
            
            @info "✓ Training successful" session=session_id
            
            return json(results)
        else
            @warn "Training failed" error=results["message"]
            return json(results)
        end
        
    catch e
        @error "Training error" exception=(e, catch_backtrace())
        return json(Dict(
            "status" => "error", 
            "message" => "Training failed: $(string(e))"
        ))
    end
end

# Results endpoint
route("/api/results/:session_id", method = GET) do
    try
        session_id = params(:session_id)
        
        if !haskey(DataStore, session_id)
            return json(Dict(
                "status" => "error", 
                "message" => "Session not found or expired"
            ))
        end
        
        if !haskey(DataStore[session_id], "results")
            return json(Dict(
                "status" => "error",
                "message" => "No results available. Please run training first."
            ))
        end
        
        # Update session timestamp
        SessionTimestamps[session_id] = now()
        
        return json(DataStore[session_id]["results"])
        
    catch e
        @error "Results error" exception=(e, catch_backtrace())
        return json(Dict(
            "status" => "error",
            "message" => "Failed to retrieve results"
        ))
    end
end

# Health check
route("/api/health", method = GET) do
    json(Dict(
        "status" => "ok",
        "message" => "Julia + Python backend running",
        "active_sessions" => length(DataStore),
        "timestamp" => now()
    ))
end

# Session info endpoint
route("/api/session/:session_id", method = GET) do
    try
        session_id = params(:session_id)
        
        if !haskey(DataStore, session_id)
            return json(Dict(
                "status" => "error",
                "message" => "Session not found"
            ))
        end
        
        session = DataStore[session_id]
        
        return json(Dict(
            "status" => "success",
            "session_id" => session_id,
            "filename" => session["original_filename"],
            "rows" => nrow(session["data"]),
            "columns" => names(session["data"]),
            "has_results" => haskey(session, "results"),
            "timestamp" => session["timestamp"]
        ))
        
    catch e
        @error "Session info error" exception=(e, catch_backtrace())
        return json(Dict("status" => "error", "message" => "Failed to get session info"))
    end
end

# Initialize
ensure_directories()

println("🚀 Julia + Python Backend starting...")
println("📊 Workflow:")
println("   1. Julia handles data upload & preprocessing")
println("   2. Python handles ML training & visualization")
println("   3. Julia serves results to React")
println("\n📁 Configuration:")
println("   - Upload directory: $(CONFIG["upload_dir"])")
println("   - Python script: $(CONFIG["python_script"])")
println("   - Max file size: $(CONFIG["max_file_size"] / 1_000_000) MB")
println("\n🌐 Server running on http://localhost:8000")

up(8000)