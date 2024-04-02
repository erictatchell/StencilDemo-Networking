//***************************************************************************************
// StencilApp.cpp by Frank Luna (C) 2015 All Rights Reserved.
//***************************************************************************************

#include "d3dApp.h"
#include <iostream>
#include "MathHelper.h"
#include "UploadBuffer.h"
#include "GeometryGenerator.h"
#include "FrameResource.h"
#include <thread>
#include <mutex>
#include "Camera.h"
#include <queue>

using Microsoft::WRL::ComPtr;
using namespace DirectX;
using namespace DirectX::PackedVector;

#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "D3D12.lib")

const int gNumFrameResources = 3;

struct Packet {

	// 0 = movement
	uint8_t packetType;

	// 1 to 3
	uint16_t playerId;

	// 0 = stationary
	// 1 = moving
	uint8_t movementState;

	// 0 = x, y
	// 1 = x, +y
	// 2 = x, -y
	// 3 = +x, y
	// 4 = -x, y
	// 5 = +x, +y
	// 6 = -x, +y
	// 7 = -x, +y
	// 8 = +x, -y
	uint8_t direction;

	uint32_t timestamp;

	std::string name;
};

// Lightweight structure stores parameters to draw a shape.  This will
// vary from app-to-app.
struct RenderItem
{
	RenderItem() = default;

	// World matrix of the shape that describes the object's local space
	// relative to the world space, which defines the position, orientation,
	// and scale of the object in the world.
	XMFLOAT4X4 World = MathHelper::Identity4x4();

	XMFLOAT4X4 TexTransform = MathHelper::Identity4x4();

	// Dirty flag indicating the object data has changed and we need to update the constant buffer.
	// Because we have an object cbuffer for each FrameResource, we have to apply the
	// update to each FrameResource.  Thus, when we modify obect data we should set 
	// NumFramesDirty = gNumFrameResources so that each frame resource gets the update.
	int NumFramesDirty = gNumFrameResources;

	// Index into GPU constant buffer corresponding to the ObjectCB for this render item.
	UINT ObjCBIndex = -1;

	Material* Mat = nullptr;
	MeshGeometry* Geo = nullptr;

	// Primitive topology.
	D3D12_PRIMITIVE_TOPOLOGY PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

	// DrawIndexedInstanced parameters.
	UINT IndexCount = 0;
	UINT StartIndexLocation = 0;
	int BaseVertexLocation = 0;
};

enum class RenderLayer : int
{
	Opaque = 0,
	Mirrors,
	Reflected,
	Transparent,
	Shadow,
	Count
};

class StencilApp : public D3DApp
{
public:
	StencilApp(HINSTANCE hInstance);
	StencilApp(const StencilApp& rhs) = delete;
	StencilApp& operator=(const StencilApp& rhs) = delete;
	~StencilApp();
	int player = 1;
	bool running;

	std::string receivedMessage;
	std::mutex messageMutex;
	std::mutex movementMutex;

	void ProcessPositionData(const GameTimer& gt);

	void OutputDebugMessage(const std::string& message);

	void SendAcknowledgement(SOCKET udpSocket);

	void StartAsyncMessageReceiver(SOCKET& udpSocket, std::atomic<bool>& isRunning);

	virtual bool Initialize()override;

private:
	virtual void OnResize()override;
	virtual void Update(const GameTimer& gt)override;
	void UpdateGameState(const GameTimer& gt);
	void SendPacket(const Packet& packet);
	void ParsePacket(const char* buf, Packet& packet);
	virtual void Draw(const GameTimer& gt)override;

	virtual void OnMouseDown(WPARAM btnState, int x, int y)override;
	virtual void OnMouseUp(WPARAM btnState, int x, int y)override;
	virtual void OnMouseMove(WPARAM btnState, int x, int y)override;

	void OnKeyboardInput(const GameTimer& gt);
	void SendSkullPositionUpdate(const XMFLOAT3& skullPosition);
	void UpdateSkullWorldMatrix(const XMFLOAT3& position, RenderItem* skullRitem, RenderItem* reflectedRitem, RenderItem* shadowedRitem);
	void UpdateCubeWorldMatrix(const XMFLOAT3& position, RenderItem* skullRitem);
	void UpdateCamera(const GameTimer& gt);
	void AnimateMaterials(const GameTimer& gt);
	void UpdateObjectCBs(const GameTimer& gt);
	void UpdatePlayers(int player, float x, float y, float z, int health);
	void ProcessMessages(const std::string& buffer);
	void UpdateMaterialCBs(const GameTimer& gt);
	void UpdateMainPassCB(const GameTimer& gt);
	void UpdateReflectedPassCB(const GameTimer& gt);

	void UpdatePosition(bool A, bool D, bool W, bool S, float dt, XMFLOAT3* position);
	void UpdatePosition(Packet packet, float dt);
	void UpdatePosition();
	void ContinuousMovement(const GameTimer& gt);
	void LoadTextures();
	void BuildRootSignature();
	void BuildDescriptorHeaps();
	void BuildShadersAndInputLayout();
	void BuildRoomGeometry();
	void BuildCubeMirrorGeometry();
	void BuildSkull2Geometry();
	void BuildSkullGeometry();
	void BuildPSOs();
	void BuildFrameResources();
	void BuildMaterials();
	void BuildRenderItems();
	void DrawRenderItems(ID3D12GraphicsCommandList* cmdList, const std::vector<RenderItem*>& ritems);

	std::array<const CD3DX12_STATIC_SAMPLER_DESC, 6> GetStaticSamplers();

private:

	XMFLOAT3 skull1 = { 0.0f, 1.0f, -10.0f };
	XMFLOAT3 skull2 = { 5.0f, 1.0f, -10.0f };
	XMFLOAT3 cube1 = { 0.0f, 1.0f, 0.0f };
	RenderItem* mymSkullRitem;
	RenderItem* mymReflectedSkullRitem;
	RenderItem* mymShadowedSkullRitem;

	Packet lastPacket;
	int movementState = 0;

	std::vector<std::unique_ptr<FrameResource>> mFrameResources;
	FrameResource* mCurrFrameResource = nullptr;
	int mCurrFrameResourceIndex = 0;

	UINT mCbvSrvDescriptorSize = 0;

	ComPtr<ID3D12RootSignature> mRootSignature = nullptr;

	ComPtr<ID3D12DescriptorHeap> mSrvDescriptorHeap = nullptr;

	std::unordered_map<std::string, std::unique_ptr<MeshGeometry>> mGeometries;
	std::unordered_map<std::string, std::unique_ptr<Material>> mMaterials;
	std::unordered_map<std::string, std::unique_ptr<Texture>> mTextures;
	std::unordered_map<std::string, ComPtr<ID3DBlob>> mShaders;
	std::unordered_map<std::string, ComPtr<ID3D12PipelineState>> mPSOs;

	std::vector<D3D12_INPUT_ELEMENT_DESC> mInputLayout;

	// Cache render items of interest.
	RenderItem* mCubeRitem = nullptr;
	RenderItem* mSkullRitem = nullptr;
	RenderItem* mReflectedSkullRitem = nullptr;
	RenderItem* mShadowedSkullRitem = nullptr;
	RenderItem* mSkullRitem_2 = nullptr;
	RenderItem* mReflectedSkullRitem_2 = nullptr;
	RenderItem* mShadowedSkullRitem_2 = nullptr;

	// List of all the render items.
	std::vector<std::unique_ptr<RenderItem>> mAllRitems;

	// Render items divided by PSO.
	std::vector<RenderItem*> mRitemLayer[(int)RenderLayer::Count];

	PassConstants mMainPassCB;
	PassConstants mReflectedPassCB;

	XMFLOAT3 mSkullTranslation = { 0.0f, 1.0f, -5.0f };
	XMFLOAT3 mSkullTranslation2 = { 0.0f, 1.0f, -5.0f };

	Camera mCamera;

	POINT mLastMousePos;
};

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance,
	PSTR cmdLine, int showCmd)
{
	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	try
	{
		StencilApp theApp(hInstance);
		if (!theApp.Initialize())
			return 0;

		return theApp.Run();
	}
	catch (DxException& e)
	{
		MessageBox(nullptr, e.ToString().c_str(), L"HR Failed", MB_OK);
		return 0;
	}
}


StencilApp::StencilApp(HINSTANCE hInstance)
	: D3DApp(hInstance)
{
}

StencilApp::~StencilApp()
{
	if (md3dDevice != nullptr)
		FlushCommandQueue();
}



void StencilApp::OutputDebugMessage(const std::string& message) {
	OutputDebugStringA((message + "\n").c_str());
}


void StencilApp::ParsePacket(const char* buf, Packet& packet) {
	std::string data(buf);
	size_t pos = 0;

	packet.packetType = static_cast<uint8_t>(std::stoi(data.substr(pos++, 1)));
	packet.playerId = static_cast<uint16_t>(std::stoi(data.substr(pos++, 1)));
	packet.movementState = static_cast<uint8_t>(std::stoi(data.substr(pos++, 1)));
	packet.direction = static_cast<uint8_t>(std::stoi(data.substr(pos++, 1)));

	size_t nameStart = pos;
	while (nameStart < data.length() && std::isdigit(data[nameStart])) {
		++nameStart;
	}
	packet.timestamp = static_cast<uint32_t>(std::stoul(data.substr(pos, nameStart - pos)));
	pos = nameStart;
}

auto packetComparator = [](const Packet& lhs, const Packet& rhs) {
	return lhs.timestamp > rhs.timestamp;
};

std::priority_queue<Packet, std::vector<Packet>, decltype(packetComparator)> packetQueue(packetComparator);
std::mutex packetQueueMutex;

void StencilApp::StartAsyncMessageReceiver(SOCKET& udpSocket, std::atomic<bool>& isRunning) {
	std::thread receiverThread([=, &udpSocket, &isRunning]() {
		char buffer[64];
		while (isRunning) {
			sockaddr_in senderAddr;
			int senderAddrSize = sizeof(senderAddr);

			int bytesReceived = recvfrom(udpSocket, buffer, sizeof(buffer), 0, (sockaddr*)&senderAddr, &senderAddrSize);
			if (bytesReceived > 0) {
				std::lock_guard<std::mutex> guard(this->movementMutex);
				ParsePacket(buffer, this->lastPacket);
				if (this->lastPacket.movementState == 0) {
					// When movementState is 0, stop moving
					this->movementState = 0;
				}
				else {
					// Update direction and keep moving
					this->movementState = 1;
				}
			}
		}
	});
	receiverThread.detach(); // Detach the thread so it runs independently
}
void ProcessPackets() {
	
}


void StencilApp::ContinuousMovement(const GameTimer& gt) {
	std::lock_guard<std::mutex> guard(this->movementMutex);
	float dt = gt.DeltaTime();
	if (this->movementState == 1) {
		this->UpdatePosition(this->lastPacket, dt);
	}
	else {
		this->UpdatePosition();
	}
}


bool StencilApp::Initialize()
{
	if (!D3DApp::Initialize())
		return false;

	// Reset the command list to prep for initialization commands.
	ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));

	// Get the increment size of a descriptor in this heap type.  This is hardware specific, 
	// so we have to query this information.
	mCbvSrvDescriptorSize = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	mCamera.SetPosition(0.0f, 2.0f, -15.0f);
	LoadTextures();
	BuildRootSignature();
	BuildDescriptorHeaps();
	BuildShadersAndInputLayout();
	BuildRoomGeometry();
	BuildSkullGeometry();
	BuildSkull2Geometry();
	BuildCubeMirrorGeometry();
	BuildMaterials();
	BuildRenderItems();
	BuildFrameResources();
	BuildPSOs();
	Connect();

	Packet packet;
	packet.packetType = 1;
	packet.playerId = id;
	packet.direction = 0;
	packet.name = name;
	packet.movementState = 0;
	auto now = std::chrono::system_clock::now();
	auto duration = now.time_since_epoch();
	packet.timestamp = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());

	SendPacket(packet);

	//UpdateCubeWorldMatrix(cube1, mCubeRitem);
	UpdateSkullWorldMatrix(skull1, mSkullRitem, mReflectedSkullRitem, mShadowedSkullRitem);
	UpdateSkullWorldMatrix(skull2, mSkullRitem_2, mReflectedSkullRitem_2, mShadowedSkullRitem_2);

	std::atomic<bool> isRunning(true);
	StartAsyncMessageReceiver(clientSocket, isRunning);

	// Execute the initialization commands.
	ThrowIfFailed(mCommandList->Close());
	ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
	mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

	// Wait until initialization is complete.
	FlushCommandQueue();
	return true;
}

void StencilApp::OnResize()
{
	D3DApp::OnResize();
	mCamera.SetLens(0.25f * MathHelper::Pi, AspectRatio(), 1.0f, 1000.0f);
}

void StencilApp::Update(const GameTimer& gt)
{
	OnKeyboardInput(gt);

	// Cycle through the circular frame resource array.
	mCurrFrameResourceIndex = (mCurrFrameResourceIndex + 1) % gNumFrameResources;
	mCurrFrameResource = mFrameResources[mCurrFrameResourceIndex].get();

	// Has the GPU finished processing the commands of the current frame resource?
	// If not, wait until the GPU has completed commands up to this fence point.
	if (mCurrFrameResource->Fence != 0 && mFence->GetCompletedValue() < mCurrFrameResource->Fence)
	{
		HANDLE eventHandle = CreateEventEx(nullptr, false, false, EVENT_ALL_ACCESS);
		ThrowIfFailed(mFence->SetEventOnCompletion(mCurrFrameResource->Fence, eventHandle));
		WaitForSingleObject(eventHandle, INFINITE);
		CloseHandle(eventHandle);
	}
	ContinuousMovement(gt);
	AnimateMaterials(gt);
	UpdateObjectCBs(gt);
	UpdateMaterialCBs(gt);
	UpdateMainPassCB(gt);
	UpdateReflectedPassCB(gt);
	ProcessMessages(receivedMessage);

	//UpdateGameState(gt);
}


void StencilApp::Draw(const GameTimer& gt)
{
	auto cmdListAlloc = mCurrFrameResource->CmdListAlloc;

	// Reuse the memory associated with command recording.
	// We can only reset when the associated command lists have finished execution on the GPU.
	ThrowIfFailed(cmdListAlloc->Reset());

	// A command list can be reset after it has been added to the command queue via ExecuteCommandList.
	// Reusing the command list reuses memory.
	ThrowIfFailed(mCommandList->Reset(cmdListAlloc.Get(), mPSOs["opaque"].Get()));

	mCommandList->RSSetViewports(1, &mScreenViewport);
	mCommandList->RSSetScissorRects(1, &mScissorRect);

	// Indicate a state transition on the resource usage.
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
		D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

	// Clear the back buffer and depth buffer.
	mCommandList->ClearRenderTargetView(CurrentBackBufferView(), (float*)&mMainPassCB.FogColor, 0, nullptr);
	mCommandList->ClearDepthStencilView(DepthStencilView(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);

	// Specify the buffers we are going to render to.
	mCommandList->OMSetRenderTargets(1, &CurrentBackBufferView(), true, &DepthStencilView());

	ID3D12DescriptorHeap* descriptorHeaps[] = { mSrvDescriptorHeap.Get() };
	mCommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

	mCommandList->SetGraphicsRootSignature(mRootSignature.Get());

	UINT passCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(PassConstants));

	// Draw opaque items--floors, walls, skull.
	auto passCB = mCurrFrameResource->PassCB->Resource();
	mCommandList->SetGraphicsRootConstantBufferView(2, passCB->GetGPUVirtualAddress());
	DrawRenderItems(mCommandList.Get(), mRitemLayer[(int)RenderLayer::Opaque]);

	// Mark the visible mirror pixels in the stencil buffer with the value 1
	mCommandList->OMSetStencilRef(1);
	mCommandList->SetPipelineState(mPSOs["markStencilMirrors"].Get());
	DrawRenderItems(mCommandList.Get(), mRitemLayer[(int)RenderLayer::Mirrors]);

	// Draw the reflection into the mirror only (only for pixels where the stencil buffer is 1).
	// Note that we must supply a different per-pass constant buffer--one with the lights reflected.
	mCommandList->SetGraphicsRootConstantBufferView(2, passCB->GetGPUVirtualAddress() + 1 * passCBByteSize);
	mCommandList->SetPipelineState(mPSOs["drawStencilReflections"].Get());
	DrawRenderItems(mCommandList.Get(), mRitemLayer[(int)RenderLayer::Reflected]);

	// Restore main pass constants and stencil ref.
	mCommandList->SetGraphicsRootConstantBufferView(2, passCB->GetGPUVirtualAddress());
	mCommandList->OMSetStencilRef(0);

	// Draw mirror with transparency so reflection blends through.
	mCommandList->SetPipelineState(mPSOs["transparent"].Get());
	DrawRenderItems(mCommandList.Get(), mRitemLayer[(int)RenderLayer::Transparent]);

	// Draw shadows for original skull
	mCommandList->SetPipelineState(mPSOs["shadow"].Get());
	DrawRenderItems(mCommandList.Get(), mRitemLayer[(int)RenderLayer::Shadow]);

	// Indicate a state transition on the resource usage.
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
		D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

	// Done recording commands.
	ThrowIfFailed(mCommandList->Close());

	// Add the command list to the queue for execution.
	ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
	mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

	// Swap the back and front buffers
	ThrowIfFailed(mSwapChain->Present(0, 0));
	mCurrBackBuffer = (mCurrBackBuffer + 1) % SwapChainBufferCount;

	// Advance the fence value to mark commands up to this fence point.
	mCurrFrameResource->Fence = ++mCurrentFence;

	// Notify the fence when the GPU completes commands up to this fence point.
	mCommandQueue->Signal(mFence.Get(), mCurrentFence);
}


void StencilApp::OnMouseDown(WPARAM btnState, int x, int y)
{
	mLastMousePos.x = x;
	mLastMousePos.y = y;

	SetCapture(mhMainWnd);
}

void StencilApp::OnMouseUp(WPARAM btnState, int x, int y)
{
	ReleaseCapture();
}

void StencilApp::OnMouseMove(WPARAM btnState, int x, int y)
{
	if ((btnState & MK_LBUTTON) != 0)
	{
		// Make each pixel correspond to a quarter of a degree.
		float dx = XMConvertToRadians(0.25f * static_cast<float>(x - mLastMousePos.x));
		float dy = XMConvertToRadians(0.25f * static_cast<float>(y - mLastMousePos.y));

		mCamera.Pitch(dy);
		mCamera.RotateY(dx);
	}

	mLastMousePos.x = x;
	mLastMousePos.y = y;
}


int GetNumberOfDecimalPoints(float number) {
	// Convert the float number to a string
	std::string strNumber = std::to_string(number);

	// Find the position of the decimal point
	size_t decimalPointPos = strNumber.find('.');
	if (decimalPointPos == std::string::npos) {
		// If no decimal point found, return 0
		return 0;
	}

	// Calculate the number of decimal points
	return strNumber.size() - decimalPointPos - 1;
}



UINT movementState = false;
uint8_t prevDirection = 0;

uint8_t DetermineDirection(BOOL A, BOOL D, BOOL W, BOOL S) {
	if (W && !A && !D && !S) return 1; // x, +y
	if (S && !A && !D && !W) return 2; // x, -y
	if (D && !W && !S && !A) return 3; // +x, y
	if (A && !W && !S && !D) return 4; // -x, y
	if (W && D) return 5; // +x, +y
	if (W && A) return 6; // -x, +y
	if (S && A) return 7; // -x, -y
	if (S && D) return 8; // +x, -y
	return 0; // stationary or undefined combination
}
bool prevA = false;
bool prevD = false;
bool prevW = false;
bool prevS = false;
bool wasMoving = false;

void StencilApp::UpdatePosition(bool A, bool D, bool W, bool S, float dt, XMFLOAT3* position) {
	XMFLOAT3* skull = (player == 1) ? &skull1 : &skull2;
	if (A) skull->x -= 1.0f * dt;
	if (D) skull->x += 1.0f * dt;
	if (W) skull->y += 1.0f * dt;
	if (S) skull->y -= 1.0f * dt;
	skull->y = MathHelper::Max(skull->y, 0.0f);
	id == 2 ? UpdateSkullWorldMatrix(*skull, mSkullRitem_2, mReflectedSkullRitem_2, mShadowedSkullRitem_2)
		: UpdateSkullWorldMatrix(*skull, mSkullRitem, mReflectedSkullRitem, mShadowedSkullRitem);
}

void StencilApp::UpdatePosition() {
	XMFLOAT3* skull = (player == 1) ? &skull1 : &skull2;
	skull->y = MathHelper::Max(skull->y, 0.0f);
	id == 2 ? UpdateSkullWorldMatrix(*skull, mSkullRitem_2, mReflectedSkullRitem_2, mShadowedSkullRitem_2)
		: UpdateSkullWorldMatrix(*skull, mSkullRitem, mReflectedSkullRitem, mShadowedSkullRitem);
}

void StencilApp::UpdatePosition(Packet packet, float dt)
{
	XMFLOAT3* skull = (packet.playerId == 1) ? &skull1 : &skull2;
	if (packet.movementState == 1 && packet.playerId != id) {
		switch (packet.direction) {
		case 0:
			break;
		case 1:
			skull->y += 1.0f * dt;
			break;
		case 2:
			skull->y -= 1.0f * dt;
			break;
		case 3:
			skull->x += 1.0f * dt;
			break;
		case 4:
			skull->x -= 1.0f * dt;
			break;
		case 5:
			skull->x += 1.0f * dt;
			skull->y += 1.0f * dt;
			break;
		case 6:
			skull->x -= 1.0f * dt;
			skull->y += 1.0f * dt;
			break;
		case 7:
			skull->x -= 1.0f * dt;
			skull->y -= 1.0f * dt;
			break;
		case 8:
			
			skull->x += 1.0f * dt;
			skull->y -= 1.0f * dt;
			break;
		}
		skull->y = MathHelper::Max(skull->y, 0.0f);
		
		packet.playerId == 2 ? UpdateSkullWorldMatrix(*skull, mSkullRitem_2, mReflectedSkullRitem_2, mShadowedSkullRitem_2)
			: UpdateSkullWorldMatrix(*skull, mSkullRitem, mReflectedSkullRitem, mShadowedSkullRitem);

	}
	this->lastPacket = packet;

}


void StencilApp::OnKeyboardInput(const GameTimer& gt)
{
	const float dt = gt.DeltaTime();

	XMFLOAT3* currentSkullPosition = (player == 1) ? &skull1 : &skull2;

	bool currentA = (GetAsyncKeyState('A') & 0x8000) != 0;
	bool currentD = (GetAsyncKeyState('D') & 0x8000) != 0;
	bool currentW = (GetAsyncKeyState('W') & 0x8000) != 0;
	bool currentS = (GetAsyncKeyState('S') & 0x8000) != 0;

	uint8_t newDirection = DetermineDirection(currentA, currentD, currentW, currentS);
	bool justStartedMoving = !wasMoving && (currentA || currentD || currentW || currentS);
	bool justStoppedMoving = wasMoving && !(currentA || currentD || currentW || currentS);
	bool directionChanged = wasMoving && (newDirection != prevDirection);

	if (justStartedMoving || directionChanged) {
		// User began moving or changed direction
		Packet movingPacket;
		movingPacket.packetType = 0; // Movement
		movingPacket.playerId = id;
		movingPacket.movementState = 1; // Moving
		movingPacket.direction = newDirection;

		auto now = std::chrono::system_clock::now();
		auto duration = now.time_since_epoch();
		movingPacket.timestamp = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());

		SendPacket(movingPacket);
	}
	else if (justStoppedMoving) {
		// User stopped moving, send a packet indicating stationary
		Packet stopMovingPacket;
		stopMovingPacket.packetType = 0; // Movement
		stopMovingPacket.playerId = id;
		stopMovingPacket.movementState = 0; // Not moving
		stopMovingPacket.direction = 0; // Stationary

		auto now = std::chrono::system_clock::now();
		auto duration = now.time_since_epoch();
		stopMovingPacket.timestamp = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());

		SendPacket(stopMovingPacket);
	}

	if (currentA || currentD || currentW || currentS) {
		UpdatePosition(currentA, currentD, currentW, currentS, dt, currentSkullPosition);
	}

	prevA = currentA;
	prevD = currentD;
	prevW = currentW;
	prevS = currentS;
	wasMoving = currentA || currentD || currentW || currentS;
	prevDirection = newDirection;

	mCamera.UpdateViewMatrix();
}

void StencilApp::SendPacket(const Packet& packet) {

	std::string buf;
	buf.append(std::to_string(packet.packetType));
	buf.append(std::to_string(packet.playerId));
	buf.append(std::to_string(packet.movementState));
	buf.append(std::to_string(packet.direction));
	buf.append(std::to_string(packet.timestamp));
	buf.append(name);

	// Assuming `socket` and `serverAddr` are defined and set up elsewhere
	SendUDPMessage(clientSocket, buf.c_str(), "192.168.1.67", 8000);
}
void StencilApp::SendSkullPositionUpdate(const XMFLOAT3& skullPosition) 
{
	//SendUDPMessage(clientSocket, msg.c_str(), "127.0.0.1", 8000);
}

void StencilApp::UpdateSkullWorldMatrix(const XMFLOAT3& position, RenderItem* skullRitem, RenderItem* reflectedRitem, RenderItem* shadowedRitem)
{
	XMMATRIX skullRotate = XMMatrixRotationY(0.5f * MathHelper::Pi);
	XMMATRIX skullScale = XMMatrixScaling(0.45f, 0.45f, 0.45f);
	XMMATRIX skullOffset = XMMatrixTranslation(position.x, position.y, position.z);
	XMMATRIX skullWorld = skullRotate * skullScale * skullOffset;

	XMVECTOR mirrorPlane = XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f); // xy plane
	XMVECTOR shadowPlane = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f); // xz plane
	XMVECTOR toMainLight = -XMLoadFloat3(&mMainPassCB.Lights[0].Direction);
	XMMATRIX R = XMMatrixReflect(mirrorPlane);
	XMMATRIX S = XMMatrixShadow(shadowPlane, toMainLight);
	XMMATRIX shadowOffsetY = XMMatrixTranslation(0.0f, 0.001f, 0.0f);

	XMStoreFloat4x4(&skullRitem->World, skullWorld);
	XMStoreFloat4x4(&reflectedRitem->World, skullWorld * R);
	XMStoreFloat4x4(&shadowedRitem->World, skullWorld * S * shadowOffsetY);

	skullRitem->NumFramesDirty = gNumFrameResources;
	reflectedRitem->NumFramesDirty = gNumFrameResources;
	shadowedRitem->NumFramesDirty = gNumFrameResources;
}

void StencilApp::UpdateCubeWorldMatrix(const XMFLOAT3& position, RenderItem* skullRitem)
{
	XMMATRIX skullRotate = XMMatrixRotationY(0.5f * MathHelper::Pi);
	XMMATRIX skullScale = XMMatrixScaling(0.45f, 0.45f, 0.45f);
	XMMATRIX skullOffset = XMMatrixTranslation(position.x, position.y, position.z);
	XMMATRIX skullWorld = skullRotate * skullScale * skullOffset;

	XMVECTOR mirrorPlane = XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f); // xy plane
	XMVECTOR shadowPlane = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f); // xz plane
	XMVECTOR toMainLight = -XMLoadFloat3(&mMainPassCB.Lights[0].Direction);
	XMMATRIX R = XMMatrixReflect(mirrorPlane);
	XMMATRIX S = XMMatrixShadow(shadowPlane, toMainLight);
	XMMATRIX shadowOffsetY = XMMatrixTranslation(0.0f, 0.001f, 0.0f);

	XMStoreFloat4x4(&skullRitem->World, skullWorld);
	//XMStoreFloat4x4(&reflectedRitem->World, skullWorld * R);
	//XMStoreFloat4x4(&shadowedRitem->World, skullWorld * S * shadowOffsetY);

	skullRitem->NumFramesDirty = gNumFrameResources;
	//reflectedRitem->NumFramesDirty = gNumFrameResources;
	//shadowedRitem->NumFramesDirty = gNumFrameResources;
}

void StencilApp::AnimateMaterials(const GameTimer& gt)
{

}

void StencilApp::UpdateObjectCBs(const GameTimer& gt)
{
	auto currObjectCB = mCurrFrameResource->ObjectCB.get();
	for (auto& e : mAllRitems)
	{
		// Only update the cbuffer data if the constants have changed.  
		// This needs to be tracked per frame resource.
		if (e->NumFramesDirty > 0)
		{
			XMMATRIX world = XMLoadFloat4x4(&e->World);
			XMMATRIX texTransform = XMLoadFloat4x4(&e->TexTransform);

			ObjectConstants objConstants;
			XMStoreFloat4x4(&objConstants.World, XMMatrixTranspose(world));
			XMStoreFloat4x4(&objConstants.TexTransform, XMMatrixTranspose(texTransform));

			currObjectCB->CopyData(e->ObjCBIndex, objConstants);

			// Next FrameResource need to be updated too.
			e->NumFramesDirty--;
		}
	}
}

void StencilApp::UpdatePlayers(int player, float x, float y, float z, int health) {
	if (player == 1) {

		UpdateSkullWorldMatrix(
			XMFLOAT3(x, y, z),
			mSkullRitem,
			mReflectedSkullRitem,
			mShadowedSkullRitem
		);
	}
	else if (player == 2) {
		UpdateSkullWorldMatrix(
			XMFLOAT3(x, y, z),
			mSkullRitem_2,
			mReflectedSkullRitem_2,
			mShadowedSkullRitem_2
		);
	}
}

void StencilApp::ProcessMessages(const std::string& buffer) {
	std::istringstream iss(buffer);
	std::string line;
	while (std::getline(iss, line)) {
		size_t start = 0;
		while ((start = line.find('%', start)) != std::string::npos) {
			// Extract the part after '%'
			size_t end = line.find('%', start + 1); // Find the start of the next client data or end of string
			std::string datagram = line.substr(start, end - start);

			size_t pos = datagram.find(';');
			std::string ip = datagram.substr(datagram.find("ip:") + 3, pos - (datagram.find("ip:") + 3));
			datagram = datagram.substr(pos + 1);

			pos = datagram.find(';');
			std::string player_str = datagram.substr(datagram.find("player:") + 7, pos - (datagram.find("player:") + 7));
			datagram = datagram.substr(pos + 1);

			pos = datagram.find(';');
			std::string name = datagram.substr(datagram.find("name:") + 5, pos - (datagram.find("name:") + 5));
			datagram = datagram.substr(pos + 1);

			pos = datagram.find(';');
			std::string health_str = datagram.substr(datagram.find("health:") + 7, pos - (datagram.find("health:") + 7));
			datagram = datagram.substr(pos + 1);

			pos = datagram.find(';');
			std::string x_str = datagram.substr(datagram.find("x:") + 2, pos - (datagram.find("x:") + 2));
			datagram = datagram.substr(pos + 1);

			pos = datagram.find(';');
			std::string y_str = datagram.substr(datagram.find("y:") + 2, pos - (datagram.find("y:") + 2));
			std::string z_str = datagram.substr(datagram.find("z:") + 2);

			int player = std::stoi(player_str);
			float x = std::stof(x_str); // Assume x_str is extracted similarly as before
			float y = std::stof(y_str); // Assume y_str is extracted similarly as before
			float z = std::stof(z_str); // Assume z_str is extracted similarly as before
			int health = std::stoi(health_str); // Assume health_str is extracted similarly as before

			if (player != this->player) {
				UpdatePlayers(player, x, y, z, health);
			}

			if (end == std::string::npos) {
				break; // Exit the loop if no more client data is found
			}
			else {
				start = end; // Set up for the next iteration to find the next client data
			}
		}
		//SendAcknowledgement();
	}
}
void StencilApp::UpdateMaterialCBs(const GameTimer& gt)
{
	auto currMaterialCB = mCurrFrameResource->MaterialCB.get();
	for (auto& e : mMaterials)
	{
		// Only update the cbuffer data if the constants have changed.  If the cbuffer
		// data changes, it needs to be updated for each FrameResource.
		Material* mat = e.second.get();
		if (mat->NumFramesDirty > 0)
		{
			XMMATRIX matTransform = XMLoadFloat4x4(&mat->MatTransform);

			MaterialConstants matConstants;
			matConstants.DiffuseAlbedo = mat->DiffuseAlbedo;
			matConstants.FresnelR0 = mat->FresnelR0;
			matConstants.Roughness = mat->Roughness;
			XMStoreFloat4x4(&matConstants.MatTransform, XMMatrixTranspose(matTransform));

			currMaterialCB->CopyData(mat->MatCBIndex, matConstants);

			// Next FrameResource need to be updated too.
			mat->NumFramesDirty--;
		}
	}
}

void StencilApp::UpdateMainPassCB(const GameTimer& gt)
{
	XMMATRIX view = mCamera.GetView();
	XMMATRIX proj = mCamera.GetProj();

	XMMATRIX viewProj = XMMatrixMultiply(view, proj);
	XMMATRIX invView = XMMatrixInverse(&XMMatrixDeterminant(view), view);
	XMMATRIX invProj = XMMatrixInverse(&XMMatrixDeterminant(proj), proj);
	XMMATRIX invViewProj = XMMatrixInverse(&XMMatrixDeterminant(viewProj), viewProj);

	XMStoreFloat4x4(&mMainPassCB.View, XMMatrixTranspose(view));
	XMStoreFloat4x4(&mMainPassCB.InvView, XMMatrixTranspose(invView));
	XMStoreFloat4x4(&mMainPassCB.Proj, XMMatrixTranspose(proj));
	XMStoreFloat4x4(&mMainPassCB.InvProj, XMMatrixTranspose(invProj));
	XMStoreFloat4x4(&mMainPassCB.ViewProj, XMMatrixTranspose(viewProj));
	XMStoreFloat4x4(&mMainPassCB.InvViewProj, XMMatrixTranspose(invViewProj));
	mMainPassCB.EyePosW = mCamera.GetPosition3f();
	mMainPassCB.RenderTargetSize = XMFLOAT2((float)mClientWidth, (float)mClientHeight);
	mMainPassCB.InvRenderTargetSize = XMFLOAT2(1.0f / mClientWidth, 1.0f / mClientHeight);
	mMainPassCB.NearZ = 1.0f;
	mMainPassCB.FarZ = 1000.0f;
	mMainPassCB.TotalTime = gt.TotalTime();
	mMainPassCB.DeltaTime = gt.DeltaTime();
	mMainPassCB.AmbientLight = { 0.25f, 0.25f, 0.35f, 1.0f };
	mMainPassCB.Lights[0].Direction = { 0.57735f, -0.57735f, 0.57735f };
	mMainPassCB.Lights[0].Strength = { 0.6f, 0.6f, 0.6f };
	mMainPassCB.Lights[1].Direction = { -0.57735f, -0.57735f, 0.57735f };
	mMainPassCB.Lights[1].Strength = { 0.3f, 0.3f, 0.3f };
	mMainPassCB.Lights[2].Direction = { 0.0f, -0.707f, -0.707f };
	mMainPassCB.Lights[2].Strength = { 0.15f, 0.15f, 0.15f };

	// Main pass stored in index 2
	auto currPassCB = mCurrFrameResource->PassCB.get();
	currPassCB->CopyData(0, mMainPassCB);
}

void StencilApp::UpdateReflectedPassCB(const GameTimer& gt)
{
	mReflectedPassCB = mMainPassCB;

	XMVECTOR mirrorPlane = XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f); // xy plane
	XMMATRIX R = XMMatrixReflect(mirrorPlane);

	// Reflect the lighting.
	for (int i = 0; i < 3; ++i)
	{
		XMVECTOR lightDir = XMLoadFloat3(&mMainPassCB.Lights[i].Direction);
		XMVECTOR reflectedLightDir = XMVector3TransformNormal(lightDir, R);
		XMStoreFloat3(&mReflectedPassCB.Lights[i].Direction, reflectedLightDir);
	}

	// Reflected pass stored in index 1
	auto currPassCB = mCurrFrameResource->PassCB.get();
	currPassCB->CopyData(1, mReflectedPassCB);
}

void StencilApp::LoadTextures()
{
	auto bricksTex = std::make_unique<Texture>();
	bricksTex->Name = "bricksTex";
	bricksTex->Filename = L"Textures/bricks3.dds";
	ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(md3dDevice.Get(),
		mCommandList.Get(), bricksTex->Filename.c_str(),
		bricksTex->Resource, bricksTex->UploadHeap));

	auto checkboardTex = std::make_unique<Texture>();
	checkboardTex->Name = "checkboardTex";
	checkboardTex->Filename = L"Textures/checkboard.dds";
	ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(md3dDevice.Get(),
		mCommandList.Get(), checkboardTex->Filename.c_str(),
		checkboardTex->Resource, checkboardTex->UploadHeap));

	auto iceTex = std::make_unique<Texture>();
	iceTex->Name = "iceTex";
	iceTex->Filename = L"Textures/ice.dds";
	ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(md3dDevice.Get(),
		mCommandList.Get(), iceTex->Filename.c_str(),
		iceTex->Resource, iceTex->UploadHeap));

	auto white1x1Tex = std::make_unique<Texture>();
	white1x1Tex->Name = "white1x1Tex";
	white1x1Tex->Filename = L"Textures/white1x1.dds";
	ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(md3dDevice.Get(),
		mCommandList.Get(), white1x1Tex->Filename.c_str(),
		white1x1Tex->Resource, white1x1Tex->UploadHeap));

	mTextures[bricksTex->Name] = std::move(bricksTex);
	mTextures[checkboardTex->Name] = std::move(checkboardTex);
	mTextures[iceTex->Name] = std::move(iceTex);
	mTextures[white1x1Tex->Name] = std::move(white1x1Tex);
}

void StencilApp::BuildRootSignature()
{
	CD3DX12_DESCRIPTOR_RANGE texTable;
	texTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);

	// Root parameter can be a table, root descriptor or root constants.
	CD3DX12_ROOT_PARAMETER slotRootParameter[4];

	// Perfomance TIP: Order from most frequent to least frequent.
	slotRootParameter[0].InitAsDescriptorTable(1, &texTable, D3D12_SHADER_VISIBILITY_PIXEL);
	slotRootParameter[1].InitAsConstantBufferView(0);
	slotRootParameter[2].InitAsConstantBufferView(1);
	slotRootParameter[3].InitAsConstantBufferView(2);

	auto staticSamplers = GetStaticSamplers();

	// A root signature is an array of root parameters.
	CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(4, slotRootParameter,
		(UINT)staticSamplers.size(), staticSamplers.data(),
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	// create a root signature with a single slot which points to a descriptor range consisting of a single constant buffer
	ComPtr<ID3DBlob> serializedRootSig = nullptr;
	ComPtr<ID3DBlob> errorBlob = nullptr;
	HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
		serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());

	if (errorBlob != nullptr)
	{
		::OutputDebugStringA((char*)errorBlob->GetBufferPointer());
	}
	ThrowIfFailed(hr);

	ThrowIfFailed(md3dDevice->CreateRootSignature(
		0,
		serializedRootSig->GetBufferPointer(),
		serializedRootSig->GetBufferSize(),
		IID_PPV_ARGS(mRootSignature.GetAddressOf())));
}

void StencilApp::BuildDescriptorHeaps()
{
	//
	// Create the SRV heap.
	//
	D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
	srvHeapDesc.NumDescriptors = 4;
	srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&mSrvDescriptorHeap)));

	//
	// Fill out the heap with actual descriptors.
	//
	CD3DX12_CPU_DESCRIPTOR_HANDLE hDescriptor(mSrvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

	auto bricksTex = mTextures["bricksTex"]->Resource;
	auto checkboardTex = mTextures["checkboardTex"]->Resource;
	auto iceTex = mTextures["iceTex"]->Resource;
	auto white1x1Tex = mTextures["white1x1Tex"]->Resource;

	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Format = bricksTex->GetDesc().Format;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = -1;
	md3dDevice->CreateShaderResourceView(bricksTex.Get(), &srvDesc, hDescriptor);

	// next descriptor
	hDescriptor.Offset(1, mCbvSrvDescriptorSize);

	srvDesc.Format = checkboardTex->GetDesc().Format;
	md3dDevice->CreateShaderResourceView(checkboardTex.Get(), &srvDesc, hDescriptor);

	// next descriptor
	hDescriptor.Offset(1, mCbvSrvDescriptorSize);

	srvDesc.Format = iceTex->GetDesc().Format;
	md3dDevice->CreateShaderResourceView(iceTex.Get(), &srvDesc, hDescriptor);

	// next descriptor
	hDescriptor.Offset(1, mCbvSrvDescriptorSize);

	srvDesc.Format = white1x1Tex->GetDesc().Format;
	md3dDevice->CreateShaderResourceView(white1x1Tex.Get(), &srvDesc, hDescriptor);
}

void StencilApp::BuildShadersAndInputLayout()
{
	const D3D_SHADER_MACRO defines[] =
	{
		"FOG", "1",
		NULL, NULL
	};

	const D3D_SHADER_MACRO alphaTestDefines[] =
	{
		"FOG", "1",
		"ALPHA_TEST", "1",
		NULL, NULL
	};

	mShaders["standardVS"] = d3dUtil::CompileShader(L"Shaders\\Default.hlsl", nullptr, "VS", "vs_5_0");
	mShaders["opaquePS"] = d3dUtil::CompileShader(L"Shaders\\Default.hlsl", defines, "PS", "ps_5_0");
	mShaders["alphaTestedPS"] = d3dUtil::CompileShader(L"Shaders\\Default.hlsl", alphaTestDefines, "PS", "ps_5_0");

	mInputLayout =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
	};
}

void StencilApp::BuildRoomGeometry()
{
	// Create and specify geometry.  For this sample we draw a floor
// and a wall with a mirror on it.  We put the floor, wall, and
// mirror geometry in one vertex buffer.
//
//   |--------------|
//   |              |
//   |----|----|----|
//   |Wall|Mirr|Wall|
//   |    | or |    |
//   /--------------/
//  /   Floor      /
// /--------------/
	float mirrorOffset = 1.5f;

	std::array<Vertex, 24> vertices =
	{
		// Floor: Observe we tile texture coordinates.
		Vertex(-7.0f, 0.0f, -15.0f, 0.0f, 1.0f, 0.0f, 0.0f, 4.0f), // 0 
		Vertex(-7.0f, 0.0f,  0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f),
		Vertex(11.0f, 0.0f,   0.0f, 0.0f, 1.0f, 0.0f, 4.0f, 0.0f),
		Vertex(11.0f, 0.0f, -15.0f, 0.0f, 1.0f, 0.0f, 4.0f, 4.0f),
	};

	std::array<std::int16_t, 6> indices =
	{
		// Floor
		0, 1, 2,
		0, 2, 3,

	};

	SubmeshGeometry floorSubmesh;
	floorSubmesh.IndexCount = 6;
	floorSubmesh.StartIndexLocation = 0;
	floorSubmesh.BaseVertexLocation = 0;

	
	const UINT vbByteSize = (UINT)vertices.size() * sizeof(Vertex);
	const UINT ibByteSize = (UINT)indices.size() * sizeof(std::uint16_t);

	auto geo = std::make_unique<MeshGeometry>();
	geo->Name = "roomGeo";

	ThrowIfFailed(D3DCreateBlob(vbByteSize, &geo->VertexBufferCPU));
	CopyMemory(geo->VertexBufferCPU->GetBufferPointer(), vertices.data(), vbByteSize);

	ThrowIfFailed(D3DCreateBlob(ibByteSize, &geo->IndexBufferCPU));
	CopyMemory(geo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

	geo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), vertices.data(), vbByteSize, geo->VertexBufferUploader);

	geo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), indices.data(), ibByteSize, geo->IndexBufferUploader);

	geo->VertexByteStride = sizeof(Vertex);
	geo->VertexBufferByteSize = vbByteSize;
	geo->IndexFormat = DXGI_FORMAT_R16_UINT;
	geo->IndexBufferByteSize = ibByteSize;

	geo->DrawArgs["floor"] = floorSubmesh;

	mGeometries[geo->Name] = std::move(geo);
}

void StencilApp::BuildCubeMirrorGeometry() {
	 
	std::vector<Vertex> vertices(24);

	// Front face
	vertices[0] = Vertex(-1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
	vertices[1] = Vertex(1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f);
	vertices[2] = Vertex(1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
	vertices[3] = Vertex(-1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f);

	// Back face
	vertices[4] = Vertex(1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f);
	vertices[5] = Vertex(-1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f);
	vertices[6] = Vertex(-1.0f, 1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f);
	vertices[7] = Vertex(1.0f, 1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f);

	// Top face
	vertices[8] = Vertex(-1.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f);
	vertices[9] = Vertex(1.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f);
	vertices[10] = Vertex(1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f);
	vertices[11] = Vertex(-1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f);

	// Bottom face
	vertices[12] = Vertex(-1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f);
	vertices[13] = Vertex(1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f);
	vertices[14] = Vertex(1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 1.0f);
	vertices[15] = Vertex(-1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f);

	// Right face
	vertices[16] = Vertex(1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
	vertices[17] = Vertex(1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	vertices[18] = Vertex(1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f);
	vertices[19] = Vertex(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);

	// Left face
	vertices[20] = Vertex(-1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
	vertices[21] = Vertex(-1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	vertices[22] = Vertex(-1.0f, 1.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f);
	vertices[23] = Vertex(-1.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f);


	// Define indices for each face (6 faces, 2 triangles per face, 3 indices per triangle)
	std::vector<std::uint16_t> indices = {
		// Front face
		0, 1, 2, 0, 2, 3,
		// Back face
		4, 5, 6, 4, 6, 7,
		// Top face
		8, 9, 10, 8, 10, 11,
		// Bottom face
		12, 13, 14, 12, 14, 15,
		// Right face
		16, 17, 18, 16, 18, 19,
		// Left face
		20, 21, 22, 20, 22, 23
	};
	// Calculate sizes of buffers
	const UINT vbByteSize = (UINT)vertices.size() * sizeof(Vertex);
	const UINT ibByteSize = (UINT)indices.size() * sizeof(std::uint16_t);

	auto geo = std::make_unique<MeshGeometry>();
	geo->Name = "cubeGeo";

	// Create and fill vertex buffer
	ThrowIfFailed(D3DCreateBlob(vbByteSize, &geo->VertexBufferCPU));
	CopyMemory(geo->VertexBufferCPU->GetBufferPointer(), vertices.data(), vbByteSize);

	// Create and fill index buffer
	ThrowIfFailed(D3DCreateBlob(ibByteSize, &geo->IndexBufferCPU));
	CopyMemory(geo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

	// Create GPU resources
	geo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), vertices.data(), vbByteSize, geo->VertexBufferUploader);
	geo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), indices.data(), ibByteSize, geo->IndexBufferUploader);

	// Set other geometry parameters
	geo->VertexByteStride = sizeof(Vertex);
	geo->VertexBufferByteSize = vbByteSize;
	geo->IndexFormat = DXGI_FORMAT_R16_UINT;
	geo->IndexBufferByteSize = ibByteSize;

	// Set up SubmeshGeometry for each face
	std::array<std::string, 6> faceNames = { "Front", "Back", "Top", "Bottom", "Right", "Left" };
	for (int i = 0; i < 6; ++i) {
		SubmeshGeometry faceSubmesh;
		faceSubmesh.IndexCount = 6; // 2 triangles * 3 indices
		faceSubmesh.StartIndexLocation = i * 6; // Each face has 6 indices
		faceSubmesh.BaseVertexLocation = 0; // All vertices are in the same buffer
		geo->DrawArgs[faceNames[i]] = faceSubmesh;
	}

	mGeometries[geo->Name] = std::move(geo);
}
void StencilApp::BuildSkull2Geometry()
{
	std::ifstream fin("Models/skull2.txt");

	if (!fin)
	{
		MessageBox(0, L"Models/skull2.txt not found.", 0, 0);
		return;
	}

	UINT vcount = 0;
	UINT tcount = 0;
	std::string ignore;

	fin >> ignore >> vcount;
	fin >> ignore >> tcount;
	fin >> ignore >> ignore >> ignore >> ignore;

	std::vector<Vertex> vertices(vcount);
	for (UINT i = 0; i < vcount; ++i)
	{
		fin >> vertices[i].Pos.x >> vertices[i].Pos.y >> vertices[i].Pos.z;
		fin >> vertices[i].Normal.x >> vertices[i].Normal.y >> vertices[i].Normal.z;

		// Model does not have texture coordinates, so just zero them out.
		vertices[i].TexC = { 0.0f, 0.0f };
	}

	fin >> ignore;
	fin >> ignore;
	fin >> ignore;

	std::vector<std::int32_t> indices(3 * tcount);
	for (UINT i = 0; i < tcount; ++i)
	{
		fin >> indices[i * 3 + 0] >> indices[i * 3 + 1] >> indices[i * 3 + 2];
	}

	fin.close();

	//
	// Pack the indices of all the meshes into one index buffer.
	//

	const UINT vbByteSize = (UINT)vertices.size() * sizeof(Vertex);

	const UINT ibByteSize = (UINT)indices.size() * sizeof(std::int32_t);

	auto geo = std::make_unique<MeshGeometry>();
	geo->Name = "skullGeo2";

	ThrowIfFailed(D3DCreateBlob(vbByteSize, &geo->VertexBufferCPU));
	CopyMemory(geo->VertexBufferCPU->GetBufferPointer(), vertices.data(), vbByteSize);

	ThrowIfFailed(D3DCreateBlob(ibByteSize, &geo->IndexBufferCPU));
	CopyMemory(geo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

	geo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), vertices.data(), vbByteSize, geo->VertexBufferUploader);

	geo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), indices.data(), ibByteSize, geo->IndexBufferUploader);

	geo->VertexByteStride = sizeof(Vertex);
	geo->VertexBufferByteSize = vbByteSize;
	geo->IndexFormat = DXGI_FORMAT_R32_UINT;
	geo->IndexBufferByteSize = ibByteSize;

	SubmeshGeometry submesh;
	submesh.IndexCount = (UINT)indices.size();
	submesh.StartIndexLocation = 0;
	submesh.BaseVertexLocation = 0;

	geo->DrawArgs["skull2"] = submesh;

	mGeometries[geo->Name] = std::move(geo);
}
void StencilApp::BuildSkullGeometry()
{
	std::ifstream fin("Models/skull.txt");

	if (!fin)
	{
		MessageBox(0, L"Models/skull.txt not found.", 0, 0);
		return;
	}

	UINT vcount = 0;
	UINT tcount = 0;
	std::string ignore;

	fin >> ignore >> vcount;
	fin >> ignore >> tcount;
	fin >> ignore >> ignore >> ignore >> ignore;

	std::vector<Vertex> vertices(vcount);
	for (UINT i = 0; i < vcount; ++i)
	{
		fin >> vertices[i].Pos.x >> vertices[i].Pos.y >> vertices[i].Pos.z;
		fin >> vertices[i].Normal.x >> vertices[i].Normal.y >> vertices[i].Normal.z;

		// Model does not have texture coordinates, so just zero them out.
		vertices[i].TexC = { 0.0f, 0.0f };
	}

	fin >> ignore;
	fin >> ignore;
	fin >> ignore;

	std::vector<std::int32_t> indices(3 * tcount);
	for (UINT i = 0; i < tcount; ++i)
	{
		fin >> indices[i * 3 + 0] >> indices[i * 3 + 1] >> indices[i * 3 + 2];
	}

	fin.close();

	//
	// Pack the indices of all the meshes into one index buffer.
	//

	const UINT vbByteSize = (UINT)vertices.size() * sizeof(Vertex);

	const UINT ibByteSize = (UINT)indices.size() * sizeof(std::int32_t);

	auto geo = std::make_unique<MeshGeometry>();
	geo->Name = "skullGeo";

	ThrowIfFailed(D3DCreateBlob(vbByteSize, &geo->VertexBufferCPU));
	CopyMemory(geo->VertexBufferCPU->GetBufferPointer(), vertices.data(), vbByteSize);

	ThrowIfFailed(D3DCreateBlob(ibByteSize, &geo->IndexBufferCPU));
	CopyMemory(geo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

	geo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), vertices.data(), vbByteSize, geo->VertexBufferUploader);

	geo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), indices.data(), ibByteSize, geo->IndexBufferUploader);

	geo->VertexByteStride = sizeof(Vertex);
	geo->VertexBufferByteSize = vbByteSize;
	geo->IndexFormat = DXGI_FORMAT_R32_UINT;
	geo->IndexBufferByteSize = ibByteSize;

	SubmeshGeometry submesh;
	submesh.IndexCount = (UINT)indices.size();
	submesh.StartIndexLocation = 0;
	submesh.BaseVertexLocation = 0;

	geo->DrawArgs["skull"] = submesh;

	mGeometries[geo->Name] = std::move(geo);
}

void StencilApp::BuildPSOs()
{
	D3D12_GRAPHICS_PIPELINE_STATE_DESC opaquePsoDesc;

	//
	// PSO for opaque objects.
	//
	ZeroMemory(&opaquePsoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
	opaquePsoDesc.InputLayout = { mInputLayout.data(), (UINT)mInputLayout.size() };
	opaquePsoDesc.pRootSignature = mRootSignature.Get();
	opaquePsoDesc.VS =
	{
		reinterpret_cast<BYTE*>(mShaders["standardVS"]->GetBufferPointer()),
		mShaders["standardVS"]->GetBufferSize()
	};
	opaquePsoDesc.PS =
	{
		reinterpret_cast<BYTE*>(mShaders["opaquePS"]->GetBufferPointer()),
		mShaders["opaquePS"]->GetBufferSize()
	};
	opaquePsoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	opaquePsoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	opaquePsoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	opaquePsoDesc.SampleMask = UINT_MAX;
	opaquePsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	opaquePsoDesc.NumRenderTargets = 1;
	opaquePsoDesc.RTVFormats[0] = mBackBufferFormat;
	opaquePsoDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;
	opaquePsoDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;
	opaquePsoDesc.DSVFormat = mDepthStencilFormat;
	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&opaquePsoDesc, IID_PPV_ARGS(&mPSOs["opaque"])));

	//
	// PSO for transparent objects
	//

	D3D12_GRAPHICS_PIPELINE_STATE_DESC transparentPsoDesc = opaquePsoDesc;

	D3D12_RENDER_TARGET_BLEND_DESC transparencyBlendDesc;
	transparencyBlendDesc.BlendEnable = true;
	transparencyBlendDesc.LogicOpEnable = false;
	transparencyBlendDesc.SrcBlend = D3D12_BLEND_SRC_ALPHA;
	transparencyBlendDesc.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
	transparencyBlendDesc.BlendOp = D3D12_BLEND_OP_ADD;
	transparencyBlendDesc.SrcBlendAlpha = D3D12_BLEND_ONE;
	transparencyBlendDesc.DestBlendAlpha = D3D12_BLEND_ZERO;
	transparencyBlendDesc.BlendOpAlpha = D3D12_BLEND_OP_ADD;
	transparencyBlendDesc.LogicOp = D3D12_LOGIC_OP_NOOP;
	transparencyBlendDesc.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

	transparentPsoDesc.BlendState.RenderTarget[0] = transparencyBlendDesc;
	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&transparentPsoDesc, IID_PPV_ARGS(&mPSOs["transparent"])));

	//
	// PSO for marking stencil mirrors.
	//

	CD3DX12_BLEND_DESC mirrorBlendState(D3D12_DEFAULT);
	mirrorBlendState.RenderTarget[0].RenderTargetWriteMask = 0;

	D3D12_DEPTH_STENCIL_DESC mirrorDSS;
	mirrorDSS.DepthEnable = true;
	mirrorDSS.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
	mirrorDSS.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
	mirrorDSS.StencilEnable = true;
	mirrorDSS.StencilReadMask = 0xff;
	mirrorDSS.StencilWriteMask = 0xff;

	mirrorDSS.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
	mirrorDSS.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
	mirrorDSS.FrontFace.StencilPassOp = D3D12_STENCIL_OP_REPLACE;
	mirrorDSS.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;

	// We are not rendering backfacing polygons, so these settings do not matter.
	mirrorDSS.BackFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
	mirrorDSS.BackFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
	mirrorDSS.BackFace.StencilPassOp = D3D12_STENCIL_OP_REPLACE;
	mirrorDSS.BackFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;

	D3D12_GRAPHICS_PIPELINE_STATE_DESC markMirrorsPsoDesc = opaquePsoDesc;
	markMirrorsPsoDesc.BlendState = mirrorBlendState;
	markMirrorsPsoDesc.DepthStencilState = mirrorDSS;
	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&markMirrorsPsoDesc, IID_PPV_ARGS(&mPSOs["markStencilMirrors"])));


	//
	//
	//

	D3D12_GRAPHICS_PIPELINE_STATE_DESC mirrorReflectionPsoDesc = opaquePsoDesc;
	mirrorReflectionPsoDesc.DepthStencilState = mirrorDSS;
	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&mirrorReflectionPsoDesc, IID_PPV_ARGS(&mPSOs["mirrorReflection"])));

	//
	// PSO for stencil reflections.
	//

	D3D12_DEPTH_STENCIL_DESC reflectionsDSS;
	reflectionsDSS.DepthEnable = true;
	reflectionsDSS.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
	reflectionsDSS.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
	reflectionsDSS.StencilEnable = true;
	reflectionsDSS.StencilReadMask = 0xff;
	reflectionsDSS.StencilWriteMask = 0xff;

	reflectionsDSS.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
	reflectionsDSS.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
	reflectionsDSS.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
	reflectionsDSS.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_EQUAL;

	// We are not rendering backfacing polygons, so these settings do not matter.
	reflectionsDSS.BackFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
	reflectionsDSS.BackFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
	reflectionsDSS.BackFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
	reflectionsDSS.BackFace.StencilFunc = D3D12_COMPARISON_FUNC_EQUAL;

	D3D12_GRAPHICS_PIPELINE_STATE_DESC drawReflectionsPsoDesc = opaquePsoDesc;
	drawReflectionsPsoDesc.DepthStencilState = reflectionsDSS;
	drawReflectionsPsoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
	drawReflectionsPsoDesc.RasterizerState.FrontCounterClockwise = true;
	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&drawReflectionsPsoDesc, IID_PPV_ARGS(&mPSOs["drawStencilReflections"])));

	//
	// PSO for shadow objects
	//

	// We are going to draw shadows with transparency, so base it off the transparency description.
	D3D12_DEPTH_STENCIL_DESC shadowDSS;
	shadowDSS.DepthEnable = true;
	shadowDSS.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
	shadowDSS.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
	shadowDSS.StencilEnable = true;
	shadowDSS.StencilReadMask = 0xff;
	shadowDSS.StencilWriteMask = 0xff;

	shadowDSS.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
	shadowDSS.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
	shadowDSS.FrontFace.StencilPassOp = D3D12_STENCIL_OP_INCR;
	shadowDSS.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_EQUAL;

	// We are not rendering backfacing polygons, so these settings do not matter.
	shadowDSS.BackFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
	shadowDSS.BackFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
	shadowDSS.BackFace.StencilPassOp = D3D12_STENCIL_OP_INCR;
	shadowDSS.BackFace.StencilFunc = D3D12_COMPARISON_FUNC_EQUAL;

	D3D12_GRAPHICS_PIPELINE_STATE_DESC shadowPsoDesc = transparentPsoDesc;
	shadowPsoDesc.DepthStencilState = shadowDSS;
	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&shadowPsoDesc, IID_PPV_ARGS(&mPSOs["shadow"])));
}

void StencilApp::BuildFrameResources()
{
	for (int i = 0; i < gNumFrameResources; ++i)
	{
		mFrameResources.push_back(std::make_unique<FrameResource>(md3dDevice.Get(),
			2, (UINT)mAllRitems.size(), (UINT)mMaterials.size()));
	}
}

void StencilApp::BuildMaterials()
{
	auto bricks = std::make_unique<Material>();
	bricks->Name = "bricks";
	bricks->MatCBIndex = 0;
	bricks->DiffuseSrvHeapIndex = 0;
	bricks->DiffuseAlbedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
	bricks->FresnelR0 = XMFLOAT3(0.05f, 0.05f, 0.05f);
	bricks->Roughness = 0.25f;

	auto checkertile = std::make_unique<Material>();
	checkertile->Name = "checkertile";
	checkertile->MatCBIndex = 1;
	checkertile->DiffuseSrvHeapIndex = 1;
	checkertile->DiffuseAlbedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
	checkertile->FresnelR0 = XMFLOAT3(0.07f, 0.07f, 0.07f);
	checkertile->Roughness = 0.3f;

	auto icemirror = std::make_unique<Material>();
	icemirror->Name = "icemirror";
	icemirror->MatCBIndex = 2;
	icemirror->DiffuseSrvHeapIndex = 2;
	icemirror->DiffuseAlbedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 0.3f);
	icemirror->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	icemirror->Roughness = 0.5f;

	auto skullMat = std::make_unique<Material>();
	skullMat->Name = "skullMat";
	skullMat->MatCBIndex = 3;
	skullMat->DiffuseSrvHeapIndex = 3;
	skullMat->DiffuseAlbedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
	skullMat->FresnelR0 = XMFLOAT3(0.05f, 0.05f, 0.05f);
	skullMat->Roughness = 0.3f;

	auto skullMat2 = std::make_unique<Material>();
	skullMat2->Name = "skullMat2";
	skullMat2->MatCBIndex = 4;
	skullMat2->DiffuseSrvHeapIndex = 3;
	skullMat2->DiffuseAlbedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
	skullMat2->FresnelR0 = XMFLOAT3(0.05f, 0.05f, 0.05f);
	skullMat2->Roughness = 0.3f;

	auto shadowMat = std::make_unique<Material>();
	shadowMat->Name = "shadowMat";
	shadowMat->MatCBIndex = 5;
	shadowMat->DiffuseSrvHeapIndex = 3;
	shadowMat->DiffuseAlbedo = XMFLOAT4(0.0f, 0.0f, 0.0f, 0.5f);
	shadowMat->FresnelR0 = XMFLOAT3(0.001f, 0.001f, 0.001f);
	shadowMat->Roughness = 0.0f;

	auto shadowMat2 = std::make_unique<Material>();
	shadowMat2->Name = "shadowMat2";
	shadowMat2->MatCBIndex = 6;
	shadowMat2->DiffuseSrvHeapIndex = 3;
	shadowMat2->DiffuseAlbedo = XMFLOAT4(0.0f, 0.0f, 0.0f, 0.5f);
	shadowMat2->FresnelR0 = XMFLOAT3(0.001f, 0.001f, 0.001f);
	shadowMat2->Roughness = 0.0f;

	auto mirrorMaterialFront = std::make_unique<Material>();
	mirrorMaterialFront->Name = "mirrorFront";
	mirrorMaterialFront->MatCBIndex = 7;
	mirrorMaterialFront->DiffuseSrvHeapIndex = 3;
	mirrorMaterialFront->DiffuseAlbedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 0.3f);
	mirrorMaterialFront->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	mirrorMaterialFront->Roughness = 0.5f;

	auto mirrorMaterialBack = std::make_unique<Material>();
	mirrorMaterialBack->Name = "mirrorBack";
	mirrorMaterialBack->MatCBIndex = 8;
	mirrorMaterialBack->DiffuseSrvHeapIndex = 3;
	mirrorMaterialBack->DiffuseAlbedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 0.3f);
	mirrorMaterialBack->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	mirrorMaterialBack->Roughness = 0.5f;

	auto mirrorMaterialLeft = std::make_unique<Material>();
	mirrorMaterialLeft->Name = "mirrorLeft";
	mirrorMaterialLeft->MatCBIndex = 9;
	mirrorMaterialLeft->DiffuseSrvHeapIndex = 3;
	mirrorMaterialLeft->DiffuseAlbedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 0.3f);
	mirrorMaterialLeft->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	mirrorMaterialLeft->Roughness = 0.5f;

	auto mirrorMaterialRight = std::make_unique<Material>();
	mirrorMaterialRight->Name = "mirrorRight";
	mirrorMaterialRight->MatCBIndex = 10;
	mirrorMaterialRight->DiffuseSrvHeapIndex = 3;
	mirrorMaterialRight->DiffuseAlbedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 0.3f);
	mirrorMaterialRight->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	mirrorMaterialRight->Roughness = 0.5f;

	auto mirrorMaterialTop = std::make_unique<Material>();
	mirrorMaterialTop->Name = "mirrorTop";
	mirrorMaterialTop->MatCBIndex = 11;
	mirrorMaterialTop->DiffuseSrvHeapIndex = 3;
	mirrorMaterialTop->DiffuseAlbedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 0.3f);
	mirrorMaterialTop->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	mirrorMaterialTop->Roughness = 0.5f;

	auto mirrorMaterialBottom = std::make_unique<Material>();
	mirrorMaterialBottom->Name = "mirrorBottom";
	mirrorMaterialBottom->MatCBIndex = 12;
	mirrorMaterialBottom->DiffuseSrvHeapIndex = 3;
	mirrorMaterialBottom->DiffuseAlbedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 0.3f);
	mirrorMaterialBottom->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
	mirrorMaterialBottom->Roughness = 0.5f;

	mMaterials["bricks"] = std::move(bricks);
	mMaterials["checkertile"] = std::move(checkertile);
	mMaterials["icemirror"] = std::move(icemirror);
	mMaterials["skullMat"] = std::move(skullMat);
	mMaterials["skullMat2"] = std::move(skullMat2);
	mMaterials["shadowMat"] = std::move(shadowMat);
	mMaterials["shadowMat2"] = std::move(shadowMat2);
	mMaterials["mirrorFront"] = std::move(mirrorMaterialFront);
	mMaterials["mirrorBack"] = std::move(mirrorMaterialBack);
	mMaterials["mirrorLeft"] = std::move(mirrorMaterialLeft);
	mMaterials["mirrorRight"] = std::move(mirrorMaterialRight);
	mMaterials["mirrorTop"] = std::move(mirrorMaterialTop);
	mMaterials["mirrorBottom"] = std::move(mirrorMaterialBottom);
}

void StencilApp::BuildRenderItems()
{
	auto floorRitem = std::make_unique<RenderItem>();
	floorRitem->World = MathHelper::Identity4x4();
	floorRitem->TexTransform = MathHelper::Identity4x4();
	floorRitem->ObjCBIndex = 0;
	floorRitem->Mat = mMaterials["checkertile"].get();
	floorRitem->Geo = mGeometries["roomGeo"].get();
	floorRitem->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	floorRitem->IndexCount = floorRitem->Geo->DrawArgs["floor"].IndexCount;
	floorRitem->StartIndexLocation = floorRitem->Geo->DrawArgs["floor"].StartIndexLocation;
	floorRitem->BaseVertexLocation = floorRitem->Geo->DrawArgs["floor"].BaseVertexLocation;
	mRitemLayer[(int)RenderLayer::Opaque].push_back(floorRitem.get());

	/*auto wallsRitem = std::make_unique<RenderItem>();
	wallsRitem->World = MathHelper::Identity4x4();
	wallsRitem->TexTransform = MathHelper::Identity4x4();
	wallsRitem->ObjCBIndex = 1;
	wallsRitem->Mat = mMaterials["bricks"].get();
	wallsRitem->Geo = mGeometries["roomGeo"].get();
	wallsRitem->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	wallsRitem->IndexCount = wallsRitem->Geo->DrawArgs["wall"].IndexCount;
	wallsRitem->StartIndexLocation = wallsRitem->Geo->DrawArgs["wall"].StartIndexLocation;
	wallsRitem->BaseVertexLocation = wallsRitem->Geo->DrawArgs["wall"].BaseVertexLocation;
	mRitemLayer[(int)RenderLayer::Opaque].push_back(wallsRitem.get());*/


	auto skullRitem = std::make_unique<RenderItem>();
	skullRitem->World = MathHelper::Identity4x4();
	skullRitem->TexTransform = MathHelper::Identity4x4();
	skullRitem->ObjCBIndex = 1;
	skullRitem->Mat = mMaterials["skullMat"].get();
	skullRitem->Geo = mGeometries["skullGeo"].get();
	skullRitem->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	skullRitem->IndexCount = skullRitem->Geo->DrawArgs["skull"].IndexCount;
	skullRitem->StartIndexLocation = skullRitem->Geo->DrawArgs["skull"].StartIndexLocation;
	skullRitem->BaseVertexLocation = skullRitem->Geo->DrawArgs["skull"].BaseVertexLocation;
	mSkullRitem = skullRitem.get();
	mRitemLayer[(int)RenderLayer::Opaque].push_back(skullRitem.get());



	// Reflected skull will have different world matrix, so it needs to be its own render item.
	auto reflectedSkullRitem = std::make_unique<RenderItem>();
	*reflectedSkullRitem = *skullRitem;
	reflectedSkullRitem->ObjCBIndex = 2;
	mReflectedSkullRitem = reflectedSkullRitem.get();
	mRitemLayer[(int)RenderLayer::Reflected].push_back(reflectedSkullRitem.get());

	// Shadowed skull will have different world matrix, so it needs to be its own render item.
	auto shadowedSkullRitem = std::make_unique<RenderItem>();
	*shadowedSkullRitem = *skullRitem;
	shadowedSkullRitem->ObjCBIndex = 3;
	shadowedSkullRitem->Mat = mMaterials["shadowMat"].get();
	mShadowedSkullRitem = shadowedSkullRitem.get();
	mRitemLayer[(int)RenderLayer::Shadow].push_back(shadowedSkullRitem.get());

	auto skullRitem2 = std::make_unique<RenderItem>();
	skullRitem2->World = MathHelper::Identity4x4();
	skullRitem2->TexTransform = MathHelper::Identity4x4();
	skullRitem2->ObjCBIndex = 4;
	skullRitem2->Mat = mMaterials["skullMat2"].get();
	skullRitem2->Geo = mGeometries["skullGeo2"].get();
	skullRitem2->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	skullRitem2->IndexCount = skullRitem2->Geo->DrawArgs["skull2"].IndexCount;
	skullRitem2->StartIndexLocation = skullRitem2->Geo->DrawArgs["skull2"].StartIndexLocation;
	skullRitem2->BaseVertexLocation = skullRitem2->Geo->DrawArgs["skull2"].BaseVertexLocation;
	mSkullRitem_2 = skullRitem2.get();
	mRitemLayer[(int)RenderLayer::Opaque].push_back(skullRitem2.get());

	auto reflectedSkullRitem2 = std::make_unique<RenderItem>();
	*reflectedSkullRitem2 = *skullRitem2;
	reflectedSkullRitem2->ObjCBIndex = 5;
	mReflectedSkullRitem_2 = reflectedSkullRitem2.get();
	mRitemLayer[(int)RenderLayer::Reflected].push_back(reflectedSkullRitem2.get());

	// Shadowed skull will have different world matrix, so it needs to be its own render item.
	auto shadowedSkullRitem2 = std::make_unique<RenderItem>();
	*shadowedSkullRitem2 = *skullRitem2;
	shadowedSkullRitem2->ObjCBIndex = 6;
	shadowedSkullRitem2->Mat = mMaterials["shadowMat2"].get();
	mShadowedSkullRitem_2 = shadowedSkullRitem2.get();
	mRitemLayer[(int)RenderLayer::Shadow].push_back(shadowedSkullRitem2.get());

	/*auto mirrorRitem = std::make_unique<RenderItem>();
	mirrorRitem->World = MathHelper::Identity4x4();
	mirrorRitem->TexTransform = MathHelper::Identity4x4();
	mirrorRitem->ObjCBIndex = 7;
	mirrorRitem->Mat = mMaterials["icemirror"].get();
	mirrorRitem->Geo = mGeometries["roomGeo"].get();
	mirrorRitem->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	mirrorRitem->IndexCount = mirrorRitem->Geo->DrawArgs["mirror"].IndexCount;
	mirrorRitem->StartIndexLocation = mirrorRitem->Geo->DrawArgs["mirror"].StartIndexLocation;
	mirrorRitem->BaseVertexLocation = mirrorRitem->Geo->DrawArgs["mirror"].BaseVertexLocation;
	mRitemLayer[(int)RenderLayer::Mirrors].push_back(mirrorRitem.get());
	mRitemLayer[(int)RenderLayer::Transparent].push_back(mirrorRitem.get());*/

	

	mAllRitems.push_back(std::move(floorRitem));
	//mAllRitems.push_back(std::move(wallsRitem));
	mAllRitems.push_back(std::move(skullRitem));
	mAllRitems.push_back(std::move(reflectedSkullRitem));
	mAllRitems.push_back(std::move(shadowedSkullRitem));
	mAllRitems.push_back(std::move(skullRitem2));
	mAllRitems.push_back(std::move(reflectedSkullRitem2));
	mAllRitems.push_back(std::move(shadowedSkullRitem2));
	//mAllRitems.push_back(std::move(mirrorRitem));

	DirectX::XMMATRIX cubeWorld = XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(0.0f, 2.0f, 0.0f);

	// Names for each face of the cube
	std::string cubeFaceNames[6] = { "Front", "Back","Right", "Left", "Top", "Bottom"  };
	UINT k = 7; // Starting Object Constant Buffer Index for cube faces

	for (int i = 0; i < 6; ++i) // Loop through all 6 faces
	{
		auto cubeFaceRitem = std::make_unique<RenderItem>();

		// Set the world matrix for this cube face render item
		XMStoreFloat4x4(&cubeFaceRitem->World, cubeWorld);
		cubeFaceRitem->ObjCBIndex = k++; // Unique constant buffer index for each face
		cubeFaceRitem->Mat = mMaterials["icemirror"].get(); // Assign material to this face
		cubeFaceRitem->Geo = mGeometries["cubeGeo"].get(); // Use the same geometry for all faces
		cubeFaceRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

		// Set the draw arguments specific to this face
		cubeFaceRitem->IndexCount = cubeFaceRitem->Geo->DrawArgs[cubeFaceNames[i]].IndexCount;
		cubeFaceRitem->StartIndexLocation = cubeFaceRitem->Geo->DrawArgs[cubeFaceNames[i]].StartIndexLocation;
		cubeFaceRitem->BaseVertexLocation = cubeFaceRitem->Geo->DrawArgs[cubeFaceNames[i]].BaseVertexLocation;

		// Add this face's render item to the appropriate render layers
		mRitemLayer[(int)RenderLayer::Transparent].push_back(cubeFaceRitem.get());
		mRitemLayer[(int)RenderLayer::Mirrors].push_back(cubeFaceRitem.get());
		mAllRitems.push_back(std::move(cubeFaceRitem));
	}
}

void StencilApp::DrawRenderItems(ID3D12GraphicsCommandList* cmdList, const std::vector<RenderItem*>& ritems)
{
	UINT objCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));
	UINT matCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(MaterialConstants));

	auto objectCB = mCurrFrameResource->ObjectCB->Resource();
	auto matCB = mCurrFrameResource->MaterialCB->Resource();

	// For each render item...
	for (size_t i = 0; i < ritems.size(); ++i)
	{
		auto ri = ritems[i];

		cmdList->IASetVertexBuffers(0, 1, &ri->Geo->VertexBufferView());
		cmdList->IASetIndexBuffer(&ri->Geo->IndexBufferView());
		cmdList->IASetPrimitiveTopology(ri->PrimitiveType);

		CD3DX12_GPU_DESCRIPTOR_HANDLE tex(mSrvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
		tex.Offset(ri->Mat->DiffuseSrvHeapIndex, mCbvSrvDescriptorSize);

		D3D12_GPU_VIRTUAL_ADDRESS objCBAddress = objectCB->GetGPUVirtualAddress() + ri->ObjCBIndex * objCBByteSize;
		D3D12_GPU_VIRTUAL_ADDRESS matCBAddress = matCB->GetGPUVirtualAddress() + ri->Mat->MatCBIndex * matCBByteSize;

		cmdList->SetGraphicsRootDescriptorTable(0, tex);
		cmdList->SetGraphicsRootConstantBufferView(1, objCBAddress);
		cmdList->SetGraphicsRootConstantBufferView(3, matCBAddress);

		cmdList->DrawIndexedInstanced(ri->IndexCount, 1, ri->StartIndexLocation, ri->BaseVertexLocation, 0);
	}
}

std::array<const CD3DX12_STATIC_SAMPLER_DESC, 6> StencilApp::GetStaticSamplers()
{
	// Applications usually only need a handful of samplers.  So just define them all up front
	// and keep them available as part of the root signature.  

	const CD3DX12_STATIC_SAMPLER_DESC pointWrap(
		0, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_POINT, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP); // addressW

	const CD3DX12_STATIC_SAMPLER_DESC pointClamp(
		1, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_POINT, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW

	const CD3DX12_STATIC_SAMPLER_DESC linearWrap(
		2, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_LINEAR, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP); // addressW

	const CD3DX12_STATIC_SAMPLER_DESC linearClamp(
		3, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_LINEAR, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW

	const CD3DX12_STATIC_SAMPLER_DESC anisotropicWrap(
		4, // shaderRegister
		D3D12_FILTER_ANISOTROPIC, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressW
		0.0f,                             // mipLODBias
		8);                               // maxAnisotropy

	const CD3DX12_STATIC_SAMPLER_DESC anisotropicClamp(
		5, // shaderRegister
		D3D12_FILTER_ANISOTROPIC, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressW
		0.0f,                              // mipLODBias
		8);                                // maxAnisotropy

	return {
		pointWrap, pointClamp,
		linearWrap, linearClamp,
		anisotropicWrap, anisotropicClamp };
}
